"""Model metadata and display labels used by the paper's analyses."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .models.registry import list_all_models

#: Pretraining-objective labels (the 11 objectives / variants of the paper).
OBJECTIVE_LABELS = {
    "mlm": "Masked Language Modeling (MLM)",
    "mlm_wwm": "Whole Word Masking (WWM)",
    "distillation_mlm": "Knowledge Distillation + MLM",
    "replaced_token_detection": "Replaced Token Detection (RTD)",
    "span_mlm": "Span Masking + Span Boundary Objective",
    "multi_task_pretraining": "Continual Multi-Task Pretraining",
    "contrastive": "Contrastive Sentence Embedding Finetuning",
    "causal_lm": "Causal Language Modeling",
    "ssm_causal": "Causal Language Modeling (State-Space)",
    "span_corruption": "Span Corruption",
    "denoising": "Denoising Autoencoding",
}

#: Architecture families as used by the layerwise (peak) analysis.
FAMILY_LABELS = {
    "encoder": "Encoder",
    "sentence_transformer": "Sentence transformer",
    "encoder_decoder": "Encoder-decoder",
    "decoder": "Decoder",
    "ssm": "SSM",
}

#: Contrastive sentence encoders. The SGI model-family table (Table 6) groups
#: these together even when the backbone is an encoder-decoder.
SENTENCE_ENCODERS = {
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "hkunlp/instructor-base",
    "sentence-transformers/sentence-t5-base",
}

#: Encoder subfamilies of Figure 4.
ENCODER_SUBFAMILY = {
    "google-bert/bert-base-uncased": "BERT-base",
    "google-bert/bert-large-uncased": "BERT-large",
    "google-bert/bert-large-uncased-whole-word-masking": "BERT-large WWM",
    "google-bert/bert-base-multilingual-cased": "mBERT",
    "distilbert/distilbert-base-uncased": "DistilBERT",
    "microsoft/deberta-v3-base": "DeBERTa",
    "answerdotai/ModernBERT-base": "ModernBERT",
    "SpanBERT/spanbert-base-cased": "SpanBERT",
    "FacebookAI/xlm-mlm-100-1280": "XLM-MLM",
    "FacebookAI/roberta-base": "RoBERTa / XLM-R / ERNIE",
    "FacebookAI/xlm-roberta-base": "RoBERTa / XLM-R / ERNIE",
    "nghuyong/ernie-2.0-base-en": "RoBERTa / XLM-R / ERNIE",
}

TASK_LABELS = {
    "classification": "Classification",
    "pair_classification": "Pair classification",
    "sts": "STS",
    "multiple_choice_qa": "Multiple-choice QA",
}

READOUT_LABELS = {
    "cls": "CLS token",
    "last_token": "Last token",
    "mean": "Mean pooled tokens",
    "sbert_pair": "Sentence-pair readout",
}

DATASET_LABELS = {
    "stanfordnlp/sst2": "SST-2",
    "SetFit/ag_news": "AG News",
    "dair-ai/emotion": "Emotion",
    "mteb/banking77": "Banking77",
    "stanfordnlp/snli": "SNLI",
    "nyu-mll/multi_nli": "MultiNLI",
    "SetFit/mrpc": "MRPC",
    "SetFit/qqp": "QQP",
    "mteb/stsbenchmark-sts": "STS-B",
    "mteb/biosses-sts": "BIOSSES",
    "allenai/ai2_arc": "ARC",
    "Rowan/hellaswag": "HellaSwag",
    "baber/piqa": "PIQA",
}


@lru_cache(maxsize=1)
def model_table() -> pd.DataFrame:
    """One row per registry model, keyed by Hugging Face id (``model_id``)."""
    rows = []
    for spec in list_all_models():
        family = str(getattr(spec.family, "value", spec.family))
        rows.append(
            {
                "model_id": spec.hf_id,
                "model_key": spec.name,
                "family": family,
                "num_layers": int(spec.num_layers),
                "objective_code": spec.pretraining_objective,
                "pretraining_objective": OBJECTIVE_LABELS.get(spec.pretraining_objective, spec.pretraining_objective),
                "selector_family": "sentence_transformer" if spec.hf_id in SENTENCE_ENCODERS else family,
                "encoder_subfamily": ENCODER_SUBFAMILY.get(spec.hf_id),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("model_id")


def attach_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join :func:`model_table` onto any frame with a ``model_id`` column."""
    return df.merge(model_table(), on="model_id", how="left")


def short_model_name(model_id: str) -> str:
    return str(model_id).split("/")[-1]
