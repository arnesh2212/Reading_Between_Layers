<div align="center">

<img src="https://img.shields.io/badge/EACL_2026-4a4a4a?style=for-the-badge" alt="EACL 2026"><img src="https://img.shields.io/badge/PAPER-6f42c1?style=for-the-badge" alt="Paper">

<h1>Reading Between the Layers:<br>Where Representation Quality Lives, and What Geometry Reveals</h1>

<hr>

<h3>A large-scale layerwise study of 25 text models, and the Stable Geometry Index (SGI) for label-free layer selection</h3>

**Arnesh Batra**<sup>1</sup> · **Aniket Khandelwal**<sup>1</sup> · **Arush Gumber**<sup>1</sup> · **Krish Thukral**<sup>2</sup> · **Kushagra Agrawal**<sup>1</sup> · **Dev Sharma**<sup>1</sup>

<sup>1</sup>Indraprastha Institute of Information Technology Delhi, India &nbsp;&nbsp; <sup>2</sup>Manipal University Jaipur, India

<!-- TODO: point the arXiv / paper badges at the final links. -->
<a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="#"><img src="https://img.shields.io/badge/Paper-PDF-8c1515" alt="Paper"></a>
<a href="https://github.com/arnesh2212/Reading_Between_Layers"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="Code"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4c9a2a" alt="License: MIT"></a>

<img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" alt="Python 3.10+">
<img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.1+">
<img src="https://img.shields.io/badge/Hydra-1.3-89b8cd" alt="Hydra 1.3">
<img src="https://img.shields.io/badge/Hugging_Face-compatible-ffcc4d?logo=huggingface&logoColor=black" alt="Hugging Face compatible">

</div>

<br>

This repository contains the code, configs and exported results for **Reading Between the Layers**. It covers two things:

1. **A layerwise study.** Linear and MLP probes are trained on every layer of **25 pretrained models** (encoders, contrastive sentence encoders, encoder–decoders, decoders and a state-space model) across **13 datasets**, 4 task families and 4 readouts. The study shows that peak layers are structured by pretraining objective, task and readout rather than sitting at the final layer.
2. **The Stable Geometry Index (SGI).** SGI is a closed-form, fully label-free layer selector with no fitted coefficients. On 383 encoder evaluation groups it improves over the last layer by **+1.20 pp** and lifts **Hit@3 from 0.389 to 0.517**, recovering about 40% of the oracle gap without labels.

## Highlights

- **Label-free layer selection.** Give SGI four geometry traces per layer (computed from unlabeled embeddings) and it returns a layer. It needs no labels, no probes and no tuning.
- **Paper numbers in about a minute on a CPU.** The exported probe and geometry tables (`results/paper`, 1.2 MB) regenerate the paper's tables and figures without a GPU.
- **Full pipeline included.** Hydra configs and a resumable sweep driver run extraction → geometry → probes for any registered Hugging Face model.
- **Regression-tested.** `tests/test_paper_results.py` pins the headline numbers (383 groups, Gain-L 1.1952 pp, Hit@3 0.5170, 617 peak-layer instances).

| Method (383 encoder groups) | Gain-L (pp) ↑ | Gap-O (pp) ↓ | Hit@1 ↑ | Hit@3 ↑ |
|---|---:|---:|---:|---:|
| **SGI (ours)** | **1.20** | **1.80** | **0.285** | **0.517** |
| Effective rank / RankMe | 0.45 | 2.55 | 0.238 | 0.428 |
| Last layer | 0.00 | 3.00 | 0.235 | 0.389 |
| Entropy | −0.29 | 3.28 | 0.198 | 0.397 |
| IsoScore | −2.07 | 5.06 | 0.159 | 0.352 |
| *Oracle* | *3.00* | *0.00* | *1.000* | *1.000* |

---

## Contents

- [Installation](#installation)
- [Quick start: SGI on your own model](#quick-start-sgi-on-your-own-model)
- [Reproducing the paper](#reproducing-the-paper)
  - [1. From the exported results (CPU, ~1 min)](#1-from-the-exported-results-cpu-1-min)
  - [2. Full pipeline from scratch (GPU)](#2-full-pipeline-from-scratch-gpu)
  - [3. Larger-model scaling probe (Table 7)](#3-larger-model-scaling-probe-table-7)
- [The Stable Geometry Index](#the-stable-geometry-index)
- [Repository layout](#repository-layout)
- [Exported result files](#exported-result-files)
- [Tests](#tests)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/arnesh2212/Reading_Between_Layers.git
cd Reading_Between_Layers

# Analysis only: SGI + reproducing every table/figure from results/ (no GPU, no torch)
pip install -e .

# Full pipeline: embedding extraction, geometry and probe training
pip install -e ".[pipeline]"          # add ",dev" for pytest + ruff, ",wandb" for W&B logging
```

Python 3.10+ is required. The paper's runs used PyTorch 2.11, Transformers 5.8, Datasets 4.8, Sentence-Transformers 5.4, Hydra 1.3 and scikit-learn 1.8 on one NVIDIA RTX A4500 (20 GB).

## Quick start: SGI on your own model

SGI needs four geometry metrics per layer: `alpha_req_r2`, `effective_rank`, `condition_number` and `twonn_intrinsic_dim`. The pipeline computes them for you, or you can compute them directly on your own pooled embeddings:

```python
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

from reading_between_layers import sgi
from reading_between_layers.geometry import compute_all_geometry

texts = [...]  # a few hundred to a few thousand *unlabeled* inputs from your task

tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModel.from_pretrained("google-bert/bert-base-uncased").eval()
with torch.no_grad():
    batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    hidden = model(**batch, output_hidden_states=True).hidden_states[1:]  # skip the embedding layer

cfg = OmegaConf.create({"metrics": {m: True for m in
                        ("alpha_req", "effective_rank", "condition_number", "twonn_intrinsic_dim")}})
rows = [{"layer_index": i, **compute_all_geometry(h[:, 0].float(), None, cfg)}   # CLS readout
        for i, h in enumerate(hidden, start=1)]

per_layer = pd.DataFrame(rows)
best_layer = sgi.select_layer(per_layer)        # latest argmax of SGI
print(best_layer, sgi.sgi_terms(per_layer)[["residual", "capacity", "stability", "depth"]])
```

`sgi.select_layers(df, group_cols)` does the same for a long table with many (model, dataset, readout) groups. The paper computes geometry on a 10% sample of the training split (seed 42).

## Reproducing the paper

### 1. From the exported results (CPU, ~1 min)

`results/paper/` holds the seed-level probe scores and per-layer geometry of the full study, exported from the pipeline cache. The following command regenerates the paper's tables (CSV, plus a Markdown `SUMMARY.md`) and figures (PDF/PNG):

```bash
bash scripts/reproduce_paper.sh                       # -> outputs/paper/{layerwise,sgi}/

# or individually
python scripts/reproduce_layerwise.py --results results/paper --out outputs/paper
python scripts/reproduce_sgi.py       --results results/paper --out outputs/paper
```

| Paper | Output (under `outputs/paper/`) |
|---|---|
| Figure 1, Tables 16–17 (peak zones by task / readout) | `layerwise/table16_task_zones.csv`, `table17_readout_zones.csv`, `figures/figure1_peak_zones.pdf` |
| Figure 2 (geometry–performance overlays) | `layerwise/figures/figure2_geometry_overlays.pdf` |
| Tables 12, 14 (scope, task × readout × probe support) | `layerwise/table12_scope.csv`, `table14_support.csv` |
| Table 18, Figure 6 (best-layer position by family) | `layerwise/table18_family_quartiles.csv`, `figures/figure6_family_best_layer.pdf` |
| Tables 19–22 (family × objective × task × readout strata) | `layerwise/table19_*.csv` … `table22_*.csv` |
| Tables 23–24, Figure 7 (matched / pairwise readouts) | `layerwise/table23_matched_readouts.csv`, `table24_pairwise_readouts.csv` |
| Table 26 (SGI coordinates, best vs. final layer) | `layerwise/table26_coordinate_deltas.csv` |
| Table 1 (encoder selector comparison) | `sgi/table1_encoder_results.csv` |
| Table 2 (paired bootstrap CI + Wilcoxon) | `sgi/table2_significance.csv` |
| Table 3 (per readout) | `sgi/table3_pooling.csv` |
| Tables 4, 6 (all model families, by task / by family) | `sgi/table4_tasks_all_families.csv`, `table6_model_families.csv` |
| Tables 5, 8 (vs. supervised all-layer scalar mix) | `sgi/table5_learned_mix.csv`, `table8_learned_mix_by_task.csv` |
| Table 9 (ablations) | `sgi/table9_ablations.csv` |
| Figures 3, 4, 5/9, 8 | `sgi/figures/figure3_*.pdf`, `figure4_*`, `figure5_*`, `figure8_*` |
| Per-group selections | `sgi/encoder_selection_results.csv`, `sgi/sgi_selected_layers.csv` |

The bootstrap in Table 2 uses 10,000 resamples (`--n-boot`). Confidence-interval endpoints can move in the second decimal with the resampling seed.

### 2. Full pipeline from scratch (GPU)

The pipeline has four stages, each a Hydra entry point under `pipeline/`:

| Stage | Script | Output in `CACHE_ROOT` |
|---|---|---|
| 01 | `01_extract_embeddings.py`: per-layer pooled representations (CLS / last token / mean / sentence-pair) | `embeddings/` |
| 02 | `02_compute_geometry.py`: geometry metrics on a 10% sample of the train split | `geometry/geometry.parquet` |
| 03 | `03_train_probes.py`: linear + MLP probes per layer, 3 seeds, plus the learned scalar mix | `probes/` |
| 04 | `04_export_results.py`: flatten the cache into `probe_scores.parquet` + `geometry.parquet` | `RESULTS_DIR` |

`scripts/run_sweep.sh` expands an experiment config into jobs and runs stages 01→03 per model/dataset combination. It skips combinations that are already complete (so it can be resumed), deletes intermediate embeddings once a combination finishes, and then runs the export.

```bash
# Smoke test: BERT-base on SST-2 (a few minutes)
EXPERIMENT=quickstart RESULTS_DIR=results/quickstart scripts/run_sweep.sh

# Main study: 25 models x 13 datasets (~66 GPU-hours on one RTX A4500)
EXPERIMENT=paper       scripts/run_sweep.sh
EXPERIMENT=paper_extra scripts/run_sweep.sh   # BERT-large/-WWM on QQP+BIOSSES, BGE on MRPC/QQP/BIOSSES

# Both runs share outputs/cache; export once more so results/paper holds everything
python pipeline/04_export_results.py --cache-root outputs/cache --out results/paper
bash scripts/reproduce_paper.sh
```

Useful knobs: `PLAN_ONLY=1` (only write the job plan), `CACHE_ROOT=...`, `DEVICE=cuda`, `DTYPE=float16`, `CONDA_ENV=<env>`, `SKIP_EXTRACT/SKIP_GEOMETRY/SKIP_PROBES/SKIP_EXPORT=1`, and `PRUNE_EMBEDDINGS_AFTER_GROUP=0` (keep embeddings on disk). The header of `scripts/run_sweep.sh` lists them all.

Individual stages can also be run by hand with Hydra overrides:

```bash
python pipeline/01_extract_embeddings.py model=bert_base_uncased dataset=sst2 task=classification \
    pooling=cls '+extraction.poolings=[cls,mean]' extraction.split=train
python pipeline/02_compute_geometry.py  model=bert_base_uncased dataset=sst2 task=classification \
    pooling=cls geometry=fast_only
python pipeline/03_train_probes.py      model=bert_base_uncased dataset=sst2 task=classification \
    pooling=cls probe=linear 'probe.seeds=[42,1337,2024]'
```

**Configs.** Models (`configs/model/`), datasets (`configs/dataset/`), tasks, probes, poolings and geometry metric sets (`configs/geometry/fast_only.yaml` is the paper's) are Hydra groups. The model and dataset registries live in `reading_between_layers/models/registry.py` and `reading_between_layers/datasets/registry.py`. Experiment configs:

| `EXPERIMENT` | What it runs |
|---|---|
| `paper` | The main study: 25 models; datasets assigned per family (the 12 encoders × 8 datasets form the 383-group SGI evaluation) |
| `paper_extra` | Extra runs that are part of the 617 peak-layer instances but not the SGI encoder evaluation |
| `scaling_encoders`, `scaling_mamba` | Larger-model scaling probe (DeBERTa-v3-large, XLM-R-large; Mamba-2.8B) |
| `quickstart` | BERT-base × SST-2 smoke test |

The two paper configs plan 620 probe groups. The analysis uses the 617 that completed in the original run: two BGE × QQP MLP probes and one XLM-MLM × Emotion CLS MLP probe are absent from `results/paper`.

### 3. Larger-model scaling probe (Table 7)

```bash
EXPERIMENT=scaling_encoders CACHE_ROOT=outputs/scaling_encoders/cache \
    RESULTS_DIR=results/scaling_encoders scripts/run_sweep.sh
EXPERIMENT=scaling_mamba    CACHE_ROOT=outputs/scaling_mamba/cache \
    RESULTS_DIR=results/scaling_mamba    scripts/run_sweep.sh
python scripts/reproduce_scaling.py --results results/scaling_encoders results/scaling_mamba --out outputs/paper
```

Each model's Gain-L is the unweighted mean over its datasets of the per-dataset mean SGI gain. Encoder and SSM runs are never pooled together. To add another checkpoint, register it in `reading_between_layers/models/registry.py`, add a `configs/model/<name>.yaml` and list it in the experiment config.

## The Stable Geometry Index

For one evaluation group with layers `ℓ = 1..L` and pooled representations `H_ℓ`, each trace is normalised within the group across its layers. `N⁺` is min–max scaling, `N⁻ = 1 − N⁺`, `R⁻` is one minus the average-rank percentile, and `L⁺` is min–max scaling after `log(1 + u − min u)`:

| Term | Formula | Geometry trace | Role |
|---|---|---|---|
| Residual | `R = N⁻(x)²` | `x` = R² of the power-law fit to the spectrum (`alpha_req_r2`) | structured deviation from a single power law |
| Capacity | `C = Z(1 + Z)`, `Z = L⁺(e)` | `e` = effective rank (exp. spectral entropy) | broad usable rank |
| Stability | `S = N⁻(κ) + R⁻(κ) + L⁺(δ)` | `κ` = condition number, `δ` = TwoNN intrinsic dimension | well-conditioned local spread |
| Depth prior | `P = N⁺(d⁴(1 − d))`, `d = ℓ/L` | layer index | weak late-middle prior (mode at d = 0.8) |

```
SGI_ℓ = R_ℓ + C_ℓ + S_ℓ + P_ℓ          selected layer = latest argmax_ℓ SGI_ℓ
```

SGI uses no labels, no probe scores and no fitted weights, and it applies the same rule to every task, readout and model. The whole implementation is `reading_between_layers/sgi.py`.

## Repository layout

```
Reading_Between_Layers/
├── reading_between_layers/        # Python package
│   ├── sgi.py                     # Stable Geometry Index (normalisers, terms, selection)
│   ├── evaluation.py              # oracles, Gain-L / Gap-O / Hit@k, baselines, paired tests
│   ├── layerwise.py               # peak-layer analysis (zones, strata, readout comparisons)
│   ├── metadata.py, plotting.py, reporting.py
│   ├── geometry/                  # effective rank, alpha-ReQ, condition number, TwoNN, IsoScore, ...
│   ├── models/  datasets/  tasks/ # registries + HF wrappers (encoder, enc-dec, decoder, SSM, sentence-transformers)
│   ├── extraction/  pooling/  probing/  caching/  utils/
├── pipeline/                      # 00 resolve sweep · 01 extract · 02 geometry · 03 probes · 04 export
├── configs/                       # Hydra configs (experiment/, model/, dataset/, task/, probe/, geometry/, ...)
├── scripts/
│   ├── run_sweep.sh               # resumable end-to-end sweep driver
│   ├── reproduce_paper.sh         # all tables + figures from results/
│   ├── reproduce_layerwise.py     # Section 5 / Appendix C
│   ├── reproduce_sgi.py           # Section 6
│   ├── reproduce_scaling.py       # Table 7
│   └── cache_status.py, prune_embedding_cache.py, clear_runtime_cache.py
├── results/paper/                 # exported probe scores + geometry of the full study (1.2 MB)
└── tests/
```

## Exported result files

**`probe_scores.parquet`** has one row per model × dataset × task × readout × probe × split strategy × layer × seed:

| column | meaning |
|---|---|
| `model_id`, `dataset_id` | Hugging Face ids |
| `task_type` | `classification`, `pair_classification`, `sts`, `multiple_choice_qa` |
| `pooling_name` | readout: `cls`, `last_token`, `mean`, `sbert_pair` |
| `representation_side` | `decoder` for encoder–decoder decoder-side states, else null |
| `probe_type`, `split_strategy` | `linear` / `nonlinear`; `official_test` or `random_60_20_20` |
| `layer_aggregation` | `single_layer`, or `learnable_weighting` for the supervised all-layer scalar mix |
| `layer_index`, `seed` | layer (1-based, embedding layer excluded); probe seed (42, 1337, 2024) |
| `primary_metric` | accuracy (Spearman correlation for STS) on the evaluation split |

**`geometry.parquet`** has one row per model × dataset × split × readout × layer and one column per metric: `alpha_req`, `alpha_req_r2`, `effective_rank`, `rankme`, `condition_number`, `twonn_intrinsic_dim`, `stable_rank`, `participation_ratio`, `representation_entropy`, `top1_variance_ratio`, `anisotropy_random_pairs`, `isoscore`, `iso_eigenvalue_ratio`, `iso_mu_viswanath`, `lidar`, `linear_cka` and others.

These tables are derived from public checkpoints and datasets. They contain aggregate metrics only, with no model weights or dataset text.

## Tests

```bash
pip install -e ".[pipeline,dev]"
pytest -m "not slow"          # unit tests + paper-number regression tests (~1-2 min, CPU)
pytest -m slow                # also downloads small checkpoints and runs real forward passes
```

## Citation

```bibtex
@misc{batra2026readingbetweenlayers,
  title  = {Reading Between the Layers: Where Representation Quality Lives, and What Geometry Reveals},
  author = {Batra, Arnesh and Khandelwal, Aniket and Gumber, Arush and Thukral, Krish and
            Agrawal, Kushagra and Sharma, Dev},
  year   = {2026},
  note   = {Code: https://github.com/arnesh2212/Reading_Between_Layers}
}
```

## License

Code is released under the [MIT License](LICENSE). The pretrained checkpoints and datasets used in the study keep their original licenses and access terms; see Appendix D of the paper.
