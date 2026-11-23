"""Batch construction for the extraction pipeline."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from ..datasets.base import DatasetSample

logger = logging.getLogger(__name__)

# Batch tokenizer calls in chunks of this many texts when measuring lengths
# for bucket-sort. Single-sample tokenizer calls are ~1000x slower per text
# than batched calls, which is what made the 120k-sample case appear to hang.
_LEN_PROBE_CHUNK = 2048


def _bucket_by_length(
    samples: list[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[DatasetSample]:
    """Sort samples by tokenized length for cheaper dynamic padding.

    Uses batch tokenization in chunks of ``_LEN_PROBE_CHUNK`` to avoid the
    per-sample Python-level tokenizer overhead.
    """
    n = len(samples)
    lengths: list[int] = []
    pbar = tqdm(
        range(0, n, _LEN_PROBE_CHUNK),
        desc="bucket-sort lengths",
        unit="chunk",
        leave=False,
    )
    for start in pbar:
        chunk = samples[start : start + _LEN_PROBE_CHUNK]
        encoded = _tokenize_text_chunk(
            chunk,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    pairs = sorted(zip(lengths, range(n), strict=True))
    return [samples[idx] for _, idx in pairs]


def build_batches(
    samples: Iterable[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    padding: str = "longest",
    sort_by_length: bool = False,
    prefetch_batches: int = 0,
    tokenizer_workers: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield tokenized batches drawn from ``samples``.

    Each batch dict contains ``input_ids``, ``attention_mask`` tensors plus a
    ``samples`` list carrying the original :class:`DatasetSample`s (the latter is
    convenient for downstream label handling).
    """
    materialized = list(samples)
    if sort_by_length and materialized:
        materialized = _bucket_by_length(materialized, tokenizer, max_length)
    chunks = (
        materialized[i : i + batch_size] for i in range(0, len(materialized), batch_size)
    )
    encode = lambda chunk: _encode_text_batch(chunk, tokenizer, max_length, padding)
    if int(prefetch_batches) <= 0:
        for chunk in chunks:
            yield encode(chunk)
        return
    yield from _prefetched(chunks, encode, prefetch_batches, tokenizer_workers)


def _encode_text_batch(
    chunk: list[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    padding: str,
) -> dict[str, Any]:
    encoded = _tokenize_text_chunk(
        chunk,
        tokenizer=tokenizer,
        max_length=max_length,
        padding=padding,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get(
            "attention_mask", torch.ones_like(encoded["input_ids"])
        ),
        "samples": chunk,
    }


def _prefetched(
    chunks: Iterable[list[DatasetSample]],
    encode_fn,
    prefetch_batches: int,
    tokenizer_workers: int,
) -> Iterator[Any]:
    """Yield encoded batches while a small rolling buffer tokenizes ahead."""
    max_pending = max(1, int(prefetch_batches))
    max_workers = max(1, int(tokenizer_workers))
    iterator = iter(chunks)
    futures: deque[Future] = deque()
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tokenize") as pool:
        for _ in range(max_pending):
            try:
                futures.append(pool.submit(encode_fn, next(iterator)))
            except StopIteration:
                break
        while futures:
            future = futures.popleft()
            batch = future.result()
            try:
                futures.append(pool.submit(encode_fn, next(iterator)))
            except StopIteration:
                pass
            yield batch


def _tokenize_text_chunk(
    chunk: list[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    padding: str | bool,
    return_tensors: str | None,
    add_special_tokens: bool = True,
) -> dict[str, Any]:
    first = [s.inputs[0] if s.inputs else "" for s in chunk]
    has_pairs = any(len(s.inputs) > 1 for s in chunk)
    kwargs = {
        "padding": padding,
        "truncation": True,
        "max_length": max_length,
        "add_special_tokens": add_special_tokens,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    if has_pairs:
        second = [s.inputs[1] if len(s.inputs) > 1 else "" for s in chunk]
        return tokenizer(first, second, **kwargs)
    return tokenizer(first, **kwargs)


def build_joint_pair_batches(
    samples: Iterable[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    padding: str = "longest",
) -> Iterator[dict[str, Any]]:
    """Encode ``[a, b]`` jointly per sample for joint pair-classification."""
    chunk: list[DatasetSample] = []
    for sample in samples:
        chunk.append(sample)
        if len(chunk) == batch_size:
            yield _encode_pair_batch(chunk, tokenizer, max_length, padding)
            chunk = []
    if chunk:
        yield _encode_pair_batch(chunk, tokenizer, max_length, padding)


def _encode_pair_batch(
    chunk: list[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    padding: str,
) -> dict[str, Any]:
    first = [s.inputs[0] for s in chunk]
    second = [s.inputs[1] if len(s.inputs) > 1 else "" for s in chunk]
    encoded = tokenizer(
        first,
        second,
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get(
            "attention_mask", torch.ones_like(encoded["input_ids"])
        ),
        "samples": chunk,
    }


def build_sbert_pair_batches(
    samples: Iterable[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    padding: str = "longest",
    prefetch_batches: int = 0,
    tokenizer_workers: int = 1,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Encode each side of a pair separately. Yields ``(left_batch, right_batch)``."""
    materialized = list(samples)
    chunks = (
        materialized[i : i + batch_size] for i in range(0, len(materialized), batch_size)
    )
    encode = lambda chunk: _encode_sbert_chunk(chunk, tokenizer, max_length, padding)
    if int(prefetch_batches) > 0:
        yield from _prefetched(chunks, encode, prefetch_batches, tokenizer_workers)
        return
    chunk: list[DatasetSample] = []
    for sample in materialized:
        chunk.append(sample)
        if len(chunk) == batch_size:
            yield _encode_sbert_chunk(chunk, tokenizer, max_length, padding)
            chunk = []
    if chunk:
        yield _encode_sbert_chunk(chunk, tokenizer, max_length, padding)


def _encode_sbert_chunk(
    chunk: list[DatasetSample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    padding: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = tokenizer(
        [s.inputs[0] for s in chunk],
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    right = tokenizer(
        [s.inputs[1] if len(s.inputs) > 1 else "" for s in chunk],
        padding=padding,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return (
        {
            "input_ids": left["input_ids"],
            "attention_mask": left.get(
                "attention_mask", torch.ones_like(left["input_ids"])
            ),
            "samples": chunk,
        },
        {
            "input_ids": right["input_ids"],
            "attention_mask": right.get(
                "attention_mask", torch.ones_like(right["input_ids"])
            ),
            "samples": chunk,
        },
    )
