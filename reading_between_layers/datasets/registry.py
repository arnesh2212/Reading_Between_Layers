"""Static registry of dataset specs for the phase 1 starter set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..tasks.compatibility import TaskType

DatasetTier = Literal["tiny", "small", "medium", "large", "xlarge"]


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one dataset in the registry."""

    name: str
    hf_id: str
    hf_subset: str | None
    task_types: list[TaskType]
    mteb_category: str | None
    language: str
    languages: list[str] | None
    splits_available: list[str]
    text_fields: list[str]
    label_field: str | None
    pair_fields: list[str] | None
    num_classes: int | None
    approx_size_total: int
    size_tier: DatasetTier
    disabled_by_default: bool = False
    is_retrieval_corpus: bool = False
    notes: str = ""
    secondary_pair_fields: list[str] | None = field(default=None)


def _spec(**kwargs: object) -> DatasetSpec:
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


_CLASSIFICATION_DATASETS: list[DatasetSpec] = [
    _spec(
        name="sst2",
        hf_id="stanfordnlp/sst2",
        hf_subset=None,
        task_types=[TaskType.CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["sentence"],
        label_field="label",
        pair_fields=None,
        num_classes=2,
        approx_size_total=70_000,
        size_tier="medium",
        notes="binary sentiment",
    ),
    _spec(
        name="ag_news",
        hf_id="SetFit/ag_news",
        hf_subset=None,
        task_types=[TaskType.CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "test"],
        text_fields=["text"],
        label_field="label",
        pair_fields=None,
        num_classes=4,
        approx_size_total=127_000,
        size_tier="medium",
        notes="also supports clustering",
    ),
    _spec(
        name="banking77",
        hf_id="mteb/banking77",
        hf_subset=None,
        task_types=[TaskType.CLASSIFICATION],
        mteb_category="Classification",
        language="en",
        languages=None,
        splits_available=["train", "test"],
        text_fields=["text"],
        label_field="label",
        pair_fields=None,
        num_classes=77,
        approx_size_total=13_000,
        size_tier="small",
        notes="also supports clustering; 77-way intent",
    ),
    _spec(
        name="emotion",
        hf_id="dair-ai/emotion",
        hf_subset=None,
        task_types=[TaskType.CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["text"],
        label_field="label",
        pair_fields=None,
        num_classes=6,
        approx_size_total=20_000,
        size_tier="small",
        notes="also supports clustering; 6-way",
    ),
]


_TOKEN_CLASSIFICATION_DATASETS: list[DatasetSpec] = [
    _spec(
        name="conll2003",
        hf_id="tner/conll2003",
        hf_subset=None,
        task_types=[TaskType.TOKEN_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["tokens"],
        label_field="ner_tags",
        pair_fields=None,
        num_classes=9,
        approx_size_total=22_000,
        size_tier="small",
        notes=(
            "NER, BIO scheme. Disabled: ships as a loading script, unsupported "
            "by datasets>=3. Swap hf_id to a parquet mirror (e.g. tner/conll2003) "
            "before re-enabling."
        ),
        disabled_by_default=False,
    ),
    _spec(
        name="universal_dependencies",
        hf_id="universal-dependencies/universal_dependencies",
        hf_subset="en_ewt",
        task_types=[TaskType.TOKEN_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["tokens"],
        label_field="upos",
        pair_fields=None,
        num_classes=17,
        approx_size_total=16_000,
        size_tier="small",
        notes=(
            "UPOS tags; en_ewt subset. Disabled: ships as a loading script, "
            "unsupported by datasets>=3. Find a parquet mirror or write a "
            "CoNLL-U loader before re-enabling."
        ),
        disabled_by_default=True,
    ),
    _spec(
        name="ontonotes5",
        hf_id="SpeedOfMagic/ontonotes_english",
        hf_subset=None,
        task_types=[TaskType.TOKEN_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["tokens"],
        label_field="ner_tags",
        pair_fields=None,
        num_classes=37,
        approx_size_total=82_000,
        size_tier="medium",
        notes="non-gated fallback to LDC OntoNotes 5",
    ),
]


_PAIR_CLASSIFICATION_DATASETS: list[DatasetSpec] = [
    _spec(
        name="snli",
        hf_id="stanfordnlp/snli",
        hf_subset=None,
        task_types=[TaskType.PAIR_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["premise", "hypothesis"],
        label_field="label",
        pair_fields=["premise", "hypothesis"],
        num_classes=3,
        approx_size_total=570_000,
        size_tier="large",
        notes="NLI",
    ),
    _spec(
        name="mnli",
        hf_id="nyu-mll/multi_nli",
        hf_subset=None,
        task_types=[TaskType.PAIR_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation_matched", "validation_mismatched"],
        text_fields=["premise", "hypothesis"],
        label_field="label",
        pair_fields=["premise", "hypothesis"],
        num_classes=3,
        approx_size_total=412_000,
        size_tier="large",
        notes="NLI cross-domain",
    ),
    _spec(
        name="mrpc",
        hf_id="SetFit/mrpc",
        hf_subset=None,
        task_types=[TaskType.PAIR_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["text1", "text2"],
        label_field="label",
        pair_fields=["text1", "text2"],
        num_classes=2,
        approx_size_total=6_000,
        size_tier="small",
        notes="paraphrase",
    ),
    _spec(
        name="qqp",
        hf_id="SetFit/qqp",
        hf_subset=None,
        task_types=[TaskType.PAIR_CLASSIFICATION],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["text1", "text2"],
        label_field="label",
        pair_fields=["text1", "text2"],
        num_classes=2,
        approx_size_total=800_000,
        size_tier="large",
        notes="paraphrase",
    ),
]


_STS_DATASETS: list[DatasetSpec] = [
    _spec(
        name="sts_b",
        hf_id="mteb/stsbenchmark-sts",
        hf_subset=None,
        task_types=[TaskType.STS],
        mteb_category="STS",
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["sentence1", "sentence2"],
        label_field="score",
        pair_fields=["sentence1", "sentence2"],
        num_classes=None,
        approx_size_total=8_628,
        size_tier="small",
        notes="gold STS",
    ),
    _spec(
        name="sick_r",
        hf_id="mteb/sickr-sts",
        hf_subset=None,
        task_types=[TaskType.STS],
        mteb_category="STS",
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["sentence1", "sentence2"],
        label_field="score",
        pair_fields=["sentence1", "sentence2"],
        num_classes=None,
        approx_size_total=10_000,
        size_tier="small",
        notes="compositional STS",
    ),
    _spec(
        name="biosses",
        hf_id="mteb/biosses-sts",
        hf_subset=None,
        task_types=[TaskType.STS],
        mteb_category="STS",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["sentence1", "sentence2"],
        label_field="score",
        pair_fields=["sentence1", "sentence2"],
        num_classes=None,
        approx_size_total=100,
        size_tier="tiny",
        notes="biomedical STS; full set",
    ),
]


_RETRIEVAL_DATASETS: list[DatasetSpec] = [
    _spec(
        name="scidocs",
        hf_id="mteb/scidocs",
        hf_subset=None,
        task_types=[TaskType.RETRIEVAL, TaskType.CLUSTERING],
        mteb_category="Retrieval",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["text"],
        label_field=None,
        pair_fields=None,
        num_classes=None,
        approx_size_total=56_000,
        size_tier="medium",
        is_retrieval_corpus=True,
        notes="scientific retrieval; also clustering",
    ),
    _spec(
        name="arguana",
        hf_id="mteb/arguana",
        hf_subset=None,
        task_types=[TaskType.RETRIEVAL],
        mteb_category="Retrieval",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["text"],
        label_field=None,
        pair_fields=None,
        num_classes=None,
        approx_size_total=11_000,
        size_tier="small",
        is_retrieval_corpus=True,
        notes="argument retrieval; small corpus",
    ),
    _spec(
        name="ask_ubuntu",
        hf_id="mteb/AskUbuntuDupQuestions",
        hf_subset=None,
        task_types=[TaskType.RETRIEVAL],
        mteb_category="Reranking",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["text"],
        label_field=None,
        pair_fields=None,
        num_classes=None,
        approx_size_total=15_000,
        size_tier="small",
        is_retrieval_corpus=True,
        notes="reranking task",
    ),
]


_CLUSTERING_DATASETS: list[DatasetSpec] = [
    _spec(
        name="arxiv_clustering_s2s",
        hf_id="mteb/arxiv-clustering-s2s",
        hf_subset=None,
        task_types=[TaskType.CLUSTERING],
        mteb_category="Clustering",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["sentences"],
        label_field="labels",
        pair_fields=None,
        num_classes=None,
        approx_size_total=100_000,
        size_tier="medium",
        notes="sentence-level arxiv clustering; 31 splits",
    ),
    _spec(
        name="medrxiv_clustering_s2s",
        hf_id="mteb/medrxiv-clustering-s2s",
        hf_subset=None,
        task_types=[TaskType.CLUSTERING],
        mteb_category="Clustering",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["sentences"],
        label_field="labels",
        pair_fields=None,
        num_classes=None,
        approx_size_total=17_000,
        size_tier="small",
        notes="biomedical sentence clustering",
    ),
    _spec(
        name="twentynewsgroups_clustering",
        hf_id="mteb/twentynewsgroups-clustering",
        hf_subset=None,
        task_types=[TaskType.CLUSTERING],
        mteb_category="Clustering",
        language="en",
        languages=None,
        splits_available=["test"],
        text_fields=["sentences"],
        label_field="labels",
        pair_fields=None,
        num_classes=None,
        approx_size_total=20_000,
        size_tier="small",
        notes="classic 20 newsgroups; 10 splits",
    ),
]


_MULTIPLE_CHOICE_QA_DATASETS: list[DatasetSpec] = [
    _spec(
        name="arc",
        hf_id="allenai/ai2_arc",
        hf_subset="ARC-Challenge",
        task_types=[TaskType.MULTIPLE_CHOICE_QA],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["question", "choices"],
        label_field="answerKey",
        pair_fields=None,
        num_classes=5,
        approx_size_total=7_800,
        size_tier="small",
        notes="scientific QA, variable-choice; represented with 5 answer slots",
    ),
    _spec(
        name="hellaswag",
        hf_id="Rowan/hellaswag",
        hf_subset=None,
        task_types=[TaskType.MULTIPLE_CHOICE_QA],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["ctx", "endings"],
        label_field="label",
        pair_fields=None,
        num_classes=4,
        approx_size_total=60_000,
        size_tier="medium",
        notes="sentence completion, 4-way",
    ),
    _spec(
        name="piqa",
        hf_id="baber/piqa",
        hf_subset=None,
        task_types=[TaskType.MULTIPLE_CHOICE_QA],
        mteb_category=None,
        language="en",
        languages=None,
        splits_available=["train", "validation", "test"],
        text_fields=["goal", "sol1", "sol2"],
        label_field="label",
        pair_fields=None,
        num_classes=2,
        approx_size_total=21_000,
        size_tier="small",
        notes="physical reasoning, 2-way",
    ),
]


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        _CLASSIFICATION_DATASETS
        + _TOKEN_CLASSIFICATION_DATASETS
        + _PAIR_CLASSIFICATION_DATASETS
        + _STS_DATASETS
        + _RETRIEVAL_DATASETS
        + _CLUSTERING_DATASETS
        + _MULTIPLE_CHOICE_QA_DATASETS
    )
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the registered spec for ``name``.

    Raises:
        KeyError: if no dataset with that short name is registered.
    """
    return DATASET_REGISTRY[name]


def list_datasets_by_task_type(task_type: TaskType) -> list[DatasetSpec]:
    """Return every dataset whose ``task_types`` includes ``task_type``."""
    return [spec for spec in DATASET_REGISTRY.values() if task_type in spec.task_types]


def list_datasets_by_language(language: str) -> list[DatasetSpec]:
    """Return every dataset whose ``language`` equals ``language``."""
    return [spec for spec in DATASET_REGISTRY.values() if spec.language == language]


def list_datasets_by_tier(tier: str) -> list[DatasetSpec]:
    """Return every dataset whose ``size_tier`` equals ``tier``."""
    return [spec for spec in DATASET_REGISTRY.values() if spec.size_tier == tier]


def list_enabled_datasets() -> list[DatasetSpec]:
    """Return every dataset that is not flagged ``disabled_by_default``."""
    return [spec for spec in DATASET_REGISTRY.values() if not spec.disabled_by_default]


def list_all_datasets() -> list[DatasetSpec]:
    """Return every registered dataset."""
    return list(DATASET_REGISTRY.values())
