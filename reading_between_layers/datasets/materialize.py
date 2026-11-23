"""HF streaming helpers that normalize rows into DatasetSample objects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from .base import DatasetSample

if TYPE_CHECKING:
    from ..tasks.compatibility import TaskType
    from .registry import DatasetSpec


def iter_hf_samples(
    dataset_spec: DatasetSpec,
    split: str,
    limit: int | None,
    task_type: TaskType | str,
    log_prefix: str,
) -> Iterator[DatasetSample]:
    """Stream a HF split and yield normalized samples, capped by emitted rows."""
    from datasets import load_dataset
    from tqdm.auto import tqdm

    print(f"{log_prefix} streaming HF dataset {dataset_spec.hf_id} split={split} limit={limit}")
    iterable = load_dataset(
        dataset_spec.hf_id,
        dataset_spec.hf_subset,
        split=split,
        streaming=True,
    )
    emitted = 0
    filtered_missing_label = 0
    pbar = tqdm(
        desc=f"stream {dataset_spec.name}",
        total=limit,
        unit="row",
        leave=False,
    )
    for row in iterable:
        for sample in samples_from_hf_row(row, dataset_spec, task_type):
            if _drop_missing_supervised_label(sample, task_type):
                filtered_missing_label += 1
                continue
            yield sample
            emitted += 1
            pbar.update(1)
            if limit is not None and emitted >= limit:
                pbar.close()
                print(f"{log_prefix} streamed {emitted} rows")
                print(
                    f"{log_prefix} filtered {filtered_missing_label} "
                    "row(s) with missing/invalid supervised labels"
                )
                return
    pbar.close()
    print(f"{log_prefix} streamed {emitted} rows")
    print(
        f"{log_prefix} filtered {filtered_missing_label} "
        "row(s) with missing/invalid supervised labels"
    )


def samples_from_hf_row(
    row: dict[str, Any],
    dataset_spec: DatasetSpec,
    task_type: TaskType | str,
) -> Iterable[DatasetSample]:
    """Normalize one raw HF row, expanding packed task formats when needed."""
    task_name = getattr(task_type, "value", str(task_type))
    if task_name == "multiple_choice_qa":
        sample = _multiple_choice_sample(row, dataset_spec)
        if sample is not None:
            yield sample
        return
    if task_name == "clustering":
        expanded = list(_expanded_clustering_samples(row, dataset_spec))
        if expanded:
            yield from expanded
            return
    yield _default_sample(row, dataset_spec)


def _default_sample(row: dict[str, Any], dataset_spec: DatasetSpec) -> DatasetSample:
    text_field = dataset_spec.text_fields[0] if dataset_spec.text_fields else "text"
    pair = dataset_spec.pair_fields or []
    if pair:
        inputs = [_stringify(row.get(pair[0], "")), _stringify(row.get(pair[1], ""))]
    else:
        inputs = [_stringify(row.get(text_field, ""))]
    label = row.get(dataset_spec.label_field) if dataset_spec.label_field else None
    return DatasetSample(inputs=inputs, label=label, extras={})


def _expanded_clustering_samples(
    row: dict[str, Any],
    dataset_spec: DatasetSpec,
) -> Iterator[DatasetSample]:
    if not dataset_spec.label_field or not dataset_spec.text_fields:
        return
    texts = row.get(dataset_spec.text_fields[0])
    labels = row.get(dataset_spec.label_field)
    if not _is_sequence(texts) or not _is_sequence(labels):
        return
    for text, label in zip(texts, labels, strict=False):
        yield DatasetSample(inputs=[_stringify(text)], label=label, extras={})


def _multiple_choice_sample(
    row: dict[str, Any],
    dataset_spec: DatasetSpec,
) -> DatasetSample | None:
    stem_field = dataset_spec.text_fields[0] if dataset_spec.text_fields else "question"
    stem = _stringify(row.get(stem_field, ""))
    choice_texts, choice_labels = _choice_texts_and_labels(row)
    if not choice_texts:
        return None
    label_raw = row.get(dataset_spec.label_field) if dataset_spec.label_field else None
    label = _choice_label_to_index(label_raw, choice_labels)
    options = "\n".join(
        f"{_choice_display_label(i, choice_labels)}. {_stringify(text)}"
        for i, text in enumerate(choice_texts)
    )
    prompt = f"{stem}\n{options}" if stem else options
    return DatasetSample(
        inputs=[prompt],
        label=label,
        extras={
            "choice_texts": [_stringify(text) for text in choice_texts],
            "choice_labels": [_stringify(label) for label in choice_labels],
        },
    )


def _choice_texts_and_labels(row: dict[str, Any]) -> tuple[list[Any], list[str]]:
    choices = row.get("choices")
    if isinstance(choices, dict):
        texts = choices.get("text") or choices.get("texts") or []
        labels = choices.get("label") or choices.get("labels") or []
        return list(texts), [_stringify(label) for label in labels]
    if isinstance(choices, list):
        if choices and isinstance(choices[0], dict):
            texts = [choice.get("text", "") for choice in choices]
            labels = [choice.get("label", str(i)) for i, choice in enumerate(choices)]
            return texts, [_stringify(label) for label in labels]
        return list(choices), [str(i) for i in range(len(choices))]
    endings = row.get("endings")
    if _is_sequence(endings):
        return list(endings), [str(i) for i in range(len(endings))]
    if "sol1" in row and "sol2" in row:
        return [row.get("sol1", ""), row.get("sol2", "")], ["0", "1"]
    return [], []


def _choice_label_to_index(label: Any, choice_labels: Sequence[str]) -> int | None:
    if label is None:
        return None
    label_text = _stringify(label).strip()
    if label_text in {"", "-1"}:
        return None
    labels = [_stringify(item).strip() for item in choice_labels]
    if label_text in labels:
        return labels.index(label_text)
    if label_text.isdigit():
        return int(label_text)
    upper = label_text.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        return ord(upper) - ord("A")
    raise ValueError(f"cannot map multiple-choice label {label!r} to an option index")


def _choice_display_label(index: int, choice_labels: Sequence[str]) -> str:
    if index < len(choice_labels) and str(choice_labels[index]).strip():
        return str(choice_labels[index]).strip()
    return chr(ord("A") + index)


def _drop_missing_supervised_label(
    sample: DatasetSample,
    task_type: TaskType | str,
) -> bool:
    task_name = getattr(task_type, "value", str(task_type))
    if task_name not in {"classification", "pair_classification", "multiple_choice_qa"}:
        return False
    return _is_missing_or_negative_label(sample.label)


def _is_missing_or_negative_label(label: Any) -> bool:
    if label is None:
        return True
    if isinstance(label, str):
        stripped = label.strip()
        if stripped == "":
            return True
        try:
            return int(stripped) < 0
        except ValueError:
            return False
    try:
        return int(label) < 0
    except (TypeError, ValueError):
        return False


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
