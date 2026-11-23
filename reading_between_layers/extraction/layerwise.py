"""Single-pass extraction over all layers with cache-aware pooling and storage."""

from __future__ import annotations

import gc
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from ..pooling.selection import forward_side_for_model
from ..utils.tensor_ops import masked_mean
from .batching import build_batches, build_sbert_pair_batches

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from ..caching.keys import EmbeddingKey
    from ..caching.store import CacheStore
    from ..datasets.base import DatasetSample
    from ..models.base import ModelWrapper
    from ..pooling.base import PoolingStrategy

logger = logging.getLogger(__name__)

_TORCH_CACHE_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}
_NUMPY_CACHE_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
}


def _cache_dtypes(config: DictConfig) -> tuple[torch.dtype, type[np.floating]]:
    cache_cfg = config.get("cache", {}) if hasattr(config, "get") else {}
    dtype_name = str(cache_cfg.get("embeddings_dtype", "float16"))
    if dtype_name not in _TORCH_CACHE_DTYPES:
        raise ValueError(f"unsupported cache.embeddings_dtype={dtype_name!r}")
    return _TORCH_CACHE_DTYPES[dtype_name], _NUMPY_CACHE_DTYPES[dtype_name]


def _tokenizer_prefetch_batches(config: DictConfig) -> int:
    extraction_cfg = config.get("extraction", {}) if hasattr(config, "get") else {}
    return max(0, int(extraction_cfg.get("prefetch_batches", 0)))


def _tokenizer_workers(config: DictConfig) -> int:
    extraction_cfg = config.get("extraction", {}) if hasattr(config, "get") else {}
    return max(1, int(extraction_cfg.get("tokenizer_workers", 1)))


def _forward_with_extraction_side(
    model: ModelWrapper,
    input_ids: Tensor,
    attention_mask: Tensor,
    include_embedding: bool,
) -> list:
    """Run the model on the representation side used for cached embeddings."""
    spec = getattr(model, "spec", None)
    which_side = forward_side_for_model(spec) if spec is not None else "auto"
    kwargs: dict[str, Any] = {
        "which_side": which_side,
        "include_embedding": include_embedding,
    }
    if which_side == "decoder":
        kwargs["decoder_input_ids"] = input_ids
        kwargs["decoder_attention_mask"] = attention_mask
    return model.forward_with_all_layers(input_ids, attention_mask, **kwargs)


def extract_all_layer_embeddings(
    model: ModelWrapper,
    samples: Iterable[DatasetSample],
    pooling: PoolingStrategy,
    config: DictConfig,
    cache: CacheStore,
    embedding_keys: dict[int, EmbeddingKey],
    return_embeddings: bool = True,
) -> dict[int, Tensor]:
    """Run model forward over ``samples``, pool per layer, save to cache.

    Skips layers whose embedding is already cached. Returns the full per-layer
    dict (loaded from cache for skipped layers, freshly computed otherwise).
    """
    materialized = list(samples)
    layers_needed = {
        idx: key for idx, key in embedding_keys.items() if not cache.has_embedding(key)
    }
    if not layers_needed:
        logger.info("all %d layers cached; loading from disk", len(embedding_keys))
        if not return_embeddings:
            return {}
        return {idx: cache.load_embedding(key) for idx, key in embedding_keys.items()}
    device = _resolve_extraction_device(model, config)
    _move_model_to_device(model, device)
    _move_pooling_to_device(pooling, device)
    logger.info(
        "extracting %d layer(s) over %d samples (%d already cached) on %s",
        len(layers_needed),
        len(materialized),
        len(embedding_keys) - len(layers_needed),
        device,
    )

    batch_size = int(config.extraction.batch_size)
    retried = False
    while True:
        try:
            if return_embeddings:
                pooled = _run_extraction(
                    model=model,
                    samples=materialized,
                    pooling=pooling,
                    config=config,
                    layers_needed=layers_needed,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                _run_extraction_to_cache(
                    model=model,
                    samples=materialized,
                    pooling=pooling,
                    config=config,
                    layers_needed=layers_needed,
                    batch_size=batch_size,
                    device=device,
                    cache=cache,
                )
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if retried:
                logger.error("OOM persisted after retry; raising")
                raise
            new_batch_size = max(1, batch_size // 2)
            logger.warning(
                "OOM at batch_size=%d; retrying at batch_size=%d", batch_size, new_batch_size
            )
            batch_size = new_batch_size
            retried = True

    if not return_embeddings:
        return {}

    results: dict[int, Tensor] = {}
    for idx, key in embedding_keys.items():
        if idx in pooled:
            chunks = pooled.pop(idx)
            torch_dtype, _ = _cache_dtypes(config)
            tensor = torch.cat(chunks, dim=0).to(torch_dtype)
            del chunks
            cache.save_embedding(key, tensor)
            results[idx] = tensor
        else:
            results[idx] = cache.load_embedding(key)
    return results


def extract_multi_pooling_layer_embeddings(
    model: ModelWrapper,
    samples: Iterable[DatasetSample],
    poolings: dict[str, PoolingStrategy],
    config: DictConfig,
    cache: CacheStore,
    embedding_keys_by_pooling: dict[str, dict[int, EmbeddingKey]],
    return_embeddings: bool = False,
) -> dict[str, dict[int, Tensor]]:
    """Extract several pooling views while sharing model forwards when possible.

    Poolings that operate on the same token sequence are computed from one
    ``forward_with_all_layers`` call per batch. ``sbert_pair`` is handled in a
    second pass because it encodes each sentence side separately.
    """
    materialized = list(samples)
    needed_by_pooling = {
        name: {idx: key for idx, key in keys.items() if not cache.has_embedding(key)}
        for name, keys in embedding_keys_by_pooling.items()
    }
    needed_by_pooling = {name: keys for name, keys in needed_by_pooling.items() if keys}
    if not needed_by_pooling:
        logger.info(
            "all %d pooling/layer embedding group(s) cached; loading from disk",
            sum(len(keys) for keys in embedding_keys_by_pooling.values()),
        )
        if not return_embeddings:
            return {}
        return {
            name: {idx: cache.load_embedding(key) for idx, key in keys.items()}
            for name, keys in embedding_keys_by_pooling.items()
        }

    device = _resolve_extraction_device(model, config)
    _move_model_to_device(model, device)
    for pooling in poolings.values():
        _move_pooling_to_device(pooling, device)
    logger.info(
        "extracting %d pooling/layer group(s) over %d samples (%d already cached) on %s",
        sum(len(keys) for keys in needed_by_pooling.values()),
        len(materialized),
        sum(len(keys) for keys in embedding_keys_by_pooling.values())
        - sum(len(keys) for keys in needed_by_pooling.values()),
        device,
    )

    batch_size = int(config.extraction.batch_size)
    retried = False
    while True:
        try:
            if return_embeddings:
                pooled = _run_multi_extraction(
                    model=model,
                    samples=materialized,
                    poolings=poolings,
                    config=config,
                    needed_by_pooling=needed_by_pooling,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                _run_multi_extraction_to_cache(
                    model=model,
                    samples=materialized,
                    poolings=poolings,
                    config=config,
                    needed_by_pooling=needed_by_pooling,
                    batch_size=batch_size,
                    device=device,
                    cache=cache,
                )
                pooled = {}
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if retried:
                logger.error("OOM persisted after retry; raising")
                raise
            new_batch_size = max(1, batch_size // 2)
            logger.warning(
                "OOM at batch_size=%d; retrying at batch_size=%d", batch_size, new_batch_size
            )
            batch_size = new_batch_size
            retried = True

    if not return_embeddings:
        return {}

    results: dict[str, dict[int, Tensor]] = {}
    for pooling_name, keys in embedding_keys_by_pooling.items():
        results[pooling_name] = {}
        for idx, key in keys.items():
            chunks = pooled.get(pooling_name, {}).pop(idx, None)
            if chunks is None:
                results[pooling_name][idx] = cache.load_embedding(key)
                continue
            torch_dtype, _ = _cache_dtypes(config)
            tensor = torch.cat(chunks, dim=0).to(torch_dtype)
            cache.save_embedding(key, tensor)
            results[pooling_name][idx] = tensor
    return results


def _run_extraction(
    model: ModelWrapper,
    samples: list[DatasetSample],
    pooling: PoolingStrategy,
    config: DictConfig,
    layers_needed: dict[int, Any],
    batch_size: int,
    device: torch.device,
) -> dict[int, list[Tensor]]:
    """Run forward over batches and collect pooled tensors per needed layer."""
    pooled_per_layer: dict[int, list[Tensor]] = {idx: [] for idx in layers_needed}
    for _, pooled_batch in _iter_pooled_batches(
        model=model,
        samples=samples,
        pooling=pooling,
        config=config,
        layers_needed=layers_needed,
        batch_size=batch_size,
        device=device,
    ):
        for layer_index, pooled in pooled_batch.items():
            pooled_per_layer[layer_index].append(pooled.detach().cpu())
    return pooled_per_layer


def _run_multi_extraction(
    model: ModelWrapper,
    samples: list[DatasetSample],
    poolings: dict[str, PoolingStrategy],
    config: DictConfig,
    needed_by_pooling: dict[str, dict[int, EmbeddingKey]],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[int, list[Tensor]]]:
    """Run multi-pooling extraction and collect CPU chunks."""
    pooled: dict[str, dict[int, list[Tensor]]] = {
        name: {idx: [] for idx in keys} for name, keys in needed_by_pooling.items()
    }
    for _, pooled_batch in _iter_multi_pooled_batches(
        model=model,
        samples=samples,
        poolings=poolings,
        config=config,
        needed_by_pooling=needed_by_pooling,
        batch_size=batch_size,
        device=device,
    ):
        for pooling_name, per_layer in pooled_batch.items():
            for layer_index, tensor in per_layer.items():
                pooled[pooling_name][layer_index].append(tensor.detach().cpu())
    return pooled


def _run_extraction_to_cache(
    model: ModelWrapper,
    samples: list[DatasetSample],
    pooling: PoolingStrategy,
    config: DictConfig,
    layers_needed: dict[int, EmbeddingKey],
    batch_size: int,
    device: torch.device,
    cache: CacheStore,
) -> None:
    """Run extraction and save disk-backed layer tensors without retaining them."""
    with tempfile.TemporaryDirectory(prefix="embedding_chunks_", dir=cache.root) as tmp:
        tmp_dir = Path(tmp)
        writers: dict[int, _LayerMemmap] = {}
        offset = 0
        torch_dtype, numpy_dtype = _cache_dtypes(config)
        for batch_rows, pooled_batch in _iter_pooled_batches(
            model=model,
            samples=samples,
            pooling=pooling,
            config=config,
            layers_needed=layers_needed,
            batch_size=batch_size,
            device=device,
        ):
            for layer_index, pooled in pooled_batch.items():
                writer = writers.get(layer_index)
                if writer is None:
                    writer = _LayerMemmap.create(
                        tmp_dir=tmp_dir,
                        layer_index=layer_index,
                        rows=len(samples),
                        cols=int(pooled.size(-1)),
                        numpy_dtype=numpy_dtype,
                        torch_dtype=torch_dtype,
                    )
                    writers[layer_index] = writer
                writer.write(offset, pooled)
            offset += batch_rows

        for layer_index, key in layers_needed.items():
            writer = writers.get(layer_index)
            if writer is None:
                raise RuntimeError(f"no embeddings were produced for layer {layer_index}")
            cache.save_embedding(key, writer.as_tensor())
            writer.close()
            del writers[layer_index]
            gc.collect()


def _run_multi_extraction_to_cache(
    model: ModelWrapper,
    samples: list[DatasetSample],
    poolings: dict[str, PoolingStrategy],
    config: DictConfig,
    needed_by_pooling: dict[str, dict[int, EmbeddingKey]],
    batch_size: int,
    device: torch.device,
    cache: CacheStore,
) -> None:
    """Run multi-pooling extraction and save each layer tensor to cache."""
    with tempfile.TemporaryDirectory(prefix="embedding_chunks_", dir=cache.root) as tmp:
        tmp_dir = Path(tmp)
        writers: dict[tuple[str, int], _LayerMemmap] = {}
        offset_by_pooling: dict[str, int] = {name: 0 for name in needed_by_pooling}
        torch_dtype, numpy_dtype = _cache_dtypes(config)

        for batch_rows, pooled_batch in _iter_multi_pooled_batches(
            model=model,
            samples=samples,
            poolings=poolings,
            config=config,
            needed_by_pooling=needed_by_pooling,
            batch_size=batch_size,
            device=device,
        ):
            for pooling_name, per_layer in pooled_batch.items():
                offset = offset_by_pooling[pooling_name]
                for layer_index, pooled in per_layer.items():
                    writer_key = (pooling_name, layer_index)
                    writer = writers.get(writer_key)
                    if writer is None:
                        writer = _LayerMemmap.create(
                            tmp_dir=tmp_dir,
                            layer_index=f"{pooling_name}_{layer_index}",
                            rows=len(samples),
                            cols=int(pooled.size(-1)),
                            numpy_dtype=numpy_dtype,
                            torch_dtype=torch_dtype,
                        )
                        writers[writer_key] = writer
                    writer.write(offset, pooled)
                offset_by_pooling[pooling_name] = offset + batch_rows

        for pooling_name, keys in needed_by_pooling.items():
            for layer_index, key in keys.items():
                writer_key = (pooling_name, layer_index)
                writer = writers.get(writer_key)
                if writer is None:
                    raise RuntimeError(
                        f"no embeddings were produced for pooling={pooling_name} "
                        f"layer={layer_index}"
                    )
                cache.save_embedding(key, writer.as_tensor())
                writer.close()
                del writers[writer_key]
                gc.collect()


def _iter_pooled_batches(
    model: ModelWrapper,
    samples: list[DatasetSample],
    pooling: PoolingStrategy,
    config: DictConfig,
    layers_needed: dict[int, Any],
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[int, dict[int, Tensor]]]:
    """Yield one dict of pooled tensors per batch."""
    if getattr(pooling, "name", "") == "sbert_pair":
        yield from _iter_sbert_pair_pooled_batches(
            model=model,
            samples=samples,
            pooling=pooling,
            config=config,
            layers_needed=layers_needed,
            batch_size=batch_size,
            device=device,
        )
        return

    max_length = int(config.extraction.max_length)
    padding = str(config.extraction.padding)
    sort_by_length = bool(config.extraction.get("sort_by_length", False))
    total_batches = max(1, (len(samples) + batch_size - 1) // batch_size)
    pbar = tqdm(
        build_batches(
            samples,
            tokenizer=model.tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            padding=padding,
            sort_by_length=sort_by_length,
            prefetch_batches=_tokenizer_prefetch_batches(config),
            tokenizer_workers=_tokenizer_workers(config),
        ),
        total=total_batches,
        desc=f"extract (bs={batch_size})",
        unit="batch",
    )
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
        attention_mask = batch["attention_mask"].to(device, non_blocking=device.type == "cuda")
        pooled_batch: dict[int, Tensor] = {}
        layer_outputs = _forward_with_extraction_side(
            model,
            input_ids,
            attention_mask,
            include_embedding=bool(config.extraction.get("include_embedding_layer", False)),
        )
        for layer_out in layer_outputs:
            if layer_out.layer_index not in layers_needed:
                continue
            with torch.no_grad():
                pooled = pooling.pool(layer_out.hidden_states, layer_out.attention_mask)
            pooled_batch[layer_out.layer_index] = pooled
        yield len(batch["samples"]), pooled_batch


def _iter_multi_pooled_batches(
    model: ModelWrapper,
    samples: list[DatasetSample],
    poolings: dict[str, PoolingStrategy],
    config: DictConfig,
    needed_by_pooling: dict[str, dict[int, EmbeddingKey]],
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[int, dict[str, dict[int, Tensor]]]]:
    """Yield pooled tensors for all requested poolings in shared-forward batches."""
    sequence_poolings = {
        name: pooling for name, pooling in poolings.items() if name != "sbert_pair"
    }
    sequence_needed = {
        name: keys for name, keys in needed_by_pooling.items() if name in sequence_poolings
    }
    if sequence_needed:
        yield from _iter_sequence_multi_pooled_batches(
            model=model,
            samples=samples,
            poolings=sequence_poolings,
            config=config,
            needed_by_pooling=sequence_needed,
            batch_size=batch_size,
            device=device,
        )
    if "sbert_pair" in needed_by_pooling:
        sbert_pooling = poolings["sbert_pair"]
        for rows, pooled_batch in _iter_sbert_pair_pooled_batches(
            model=model,
            samples=samples,
            pooling=sbert_pooling,
            config=config,
            layers_needed=needed_by_pooling["sbert_pair"],
            batch_size=batch_size,
            device=device,
        ):
            yield rows, {"sbert_pair": pooled_batch}


def _iter_sequence_multi_pooled_batches(
    model: ModelWrapper,
    samples: list[DatasetSample],
    poolings: dict[str, PoolingStrategy],
    config: DictConfig,
    needed_by_pooling: dict[str, dict[int, EmbeddingKey]],
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[int, dict[str, dict[int, Tensor]]]]:
    """Yield one shared-forward batch for all sequence-level poolings."""
    max_length = int(config.extraction.max_length)
    padding = str(config.extraction.padding)
    sort_by_length = bool(config.extraction.get("sort_by_length", False))
    total_batches = max(1, (len(samples) + batch_size - 1) // batch_size)
    needed_layers = set().union(*(set(keys) for keys in needed_by_pooling.values()))
    pbar = tqdm(
        build_batches(
            samples,
            tokenizer=model.tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            padding=padding,
            sort_by_length=sort_by_length,
            prefetch_batches=_tokenizer_prefetch_batches(config),
            tokenizer_workers=_tokenizer_workers(config),
        ),
        total=total_batches,
        desc=f"extract multi (bs={batch_size})",
        unit="batch",
    )
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
        attention_mask = batch["attention_mask"].to(device, non_blocking=device.type == "cuda")
        pooled_batch: dict[str, dict[int, Tensor]] = {
            name: {} for name in needed_by_pooling
        }
        layer_outputs = _forward_with_extraction_side(
            model,
            input_ids,
            attention_mask,
            include_embedding=bool(config.extraction.get("include_embedding_layer", False)),
        )
        for layer_out in layer_outputs:
            if layer_out.layer_index not in needed_layers:
                continue
            with torch.no_grad():
                for pooling_name, pooling in poolings.items():
                    if layer_out.layer_index not in needed_by_pooling.get(pooling_name, {}):
                        continue
                    pooled_batch[pooling_name][layer_out.layer_index] = pooling.pool(
                        layer_out.hidden_states,
                        layer_out.attention_mask,
                    )
        yield len(batch["samples"]), pooled_batch


class _LayerMemmap:
    """Disk-backed temporary layer tensor used by save-only extraction."""

    def __init__(self, array: np.memmap, torch_dtype: torch.dtype) -> None:
        self.array = array
        self.torch_dtype = torch_dtype

    @classmethod
    def create(
        cls,
        tmp_dir: Path,
        layer_index: int | str,
        rows: int,
        cols: int,
        numpy_dtype: type[np.floating],
        torch_dtype: torch.dtype,
    ) -> _LayerMemmap:
        path = tmp_dir / f"layer_{layer_index}.dat"
        array = np.memmap(path, dtype=numpy_dtype, mode="w+", shape=(rows, cols))
        return cls(array, torch_dtype=torch_dtype)

    def write(self, offset: int, tensor: Tensor) -> None:
        cpu = tensor.detach().to(self.torch_dtype).cpu().contiguous()
        rows = int(cpu.size(0))
        self.array[offset : offset + rows] = cpu.numpy()
        del cpu

    def as_tensor(self) -> Tensor:
        self.array.flush()
        return torch.from_numpy(self.array)

    def close(self) -> None:
        mmap = getattr(self.array, "_mmap", None)
        del self.array
        if mmap is not None:
            mmap.close()


def _iter_sbert_pair_pooled_batches(
    model: ModelWrapper,
    samples: list[DatasetSample],
    pooling: PoolingStrategy,
    config: DictConfig,
    layers_needed: dict[int, Any],
    batch_size: int,
    device: torch.device,
) -> Iterable[tuple[int, dict[int, Tensor]]]:
    """Encode pair sides separately and combine mean-pooled side embeddings."""
    max_length = int(config.extraction.max_length)
    padding = str(config.extraction.padding)
    total_batches = max(1, (len(samples) + batch_size - 1) // batch_size)
    pbar = tqdm(
        build_sbert_pair_batches(
            samples,
            tokenizer=model.tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            padding=padding,
            prefetch_batches=_tokenizer_prefetch_batches(config),
            tokenizer_workers=_tokenizer_workers(config),
        ),
        total=total_batches,
        desc=f"extract sbert_pair (bs={batch_size})",
        unit="batch",
    )
    combine = pooling.combine  # type: ignore[attr-defined]
    for left_batch, right_batch in pbar:
        left_ids = left_batch["input_ids"].to(device, non_blocking=device.type == "cuda")
        left_mask = left_batch["attention_mask"].to(device, non_blocking=device.type == "cuda")
        right_ids = right_batch["input_ids"].to(device, non_blocking=device.type == "cuda")
        right_mask = right_batch["attention_mask"].to(device, non_blocking=device.type == "cuda")
        include_embedding = bool(config.extraction.get("include_embedding_layer", False))
        left_outputs = {
            layer.layer_index: layer
            for layer in _forward_with_extraction_side(
                model,
                left_ids,
                left_mask,
                include_embedding=include_embedding,
            )
        }
        right_outputs = {
            layer.layer_index: layer
            for layer in _forward_with_extraction_side(
                model,
                right_ids,
                right_mask,
                include_embedding=include_embedding,
            )
        }
        pooled_batch: dict[int, Tensor] = {}
        for layer_index in layers_needed:
            left_layer = left_outputs.get(layer_index)
            right_layer = right_outputs.get(layer_index)
            if left_layer is None or right_layer is None:
                continue
            with torch.no_grad():
                left_pooled = masked_mean(
                    left_layer.hidden_states,
                    left_layer.attention_mask,
                    dim=1,
                )
                right_pooled = masked_mean(
                    right_layer.hidden_states,
                    right_layer.attention_mask,
                    dim=1,
                )
                paired = combine(left_pooled, right_pooled)
            pooled_batch[layer_index] = paired
        yield len(left_batch["samples"]), pooled_batch


def _resolve_extraction_device(model: ModelWrapper, config: DictConfig) -> torch.device:
    requested = config.get("device") if hasattr(config, "get") else None
    if requested is None:
        return _model_device(model)
    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg.device is cuda, but torch.cuda.is_available() is False")
    return device


def _model_device(model: ModelWrapper) -> torch.device:
    module = getattr(model, "model", None)
    if module is None:
        return torch.device("cpu")
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_model_to_device(model: ModelWrapper, device: torch.device) -> None:
    st_model = getattr(model, "st_model", None)
    if st_model is not None:
        st_model.to(device)
        st_model.eval()
        return

    module = getattr(model, "model", None)
    if module is not None:
        module.to(device)
        module.eval()


def _move_pooling_to_device(pooling: PoolingStrategy, device: torch.device) -> None:
    to_device = getattr(pooling, "to", None)
    if callable(to_device):
        to_device(device)
    eval_mode = getattr(pooling, "eval", None)
    if callable(eval_mode):
        eval_mode()
