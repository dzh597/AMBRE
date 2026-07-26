# AMBRE: Boundary-Preserving Meta-Path and Non-Backtracking Structural Transfer for Multi-Knowledge-Graph Completion

This repository provides an anonymous, compact implementation of **AMBRE: Boundary-Preserving Meta-Path and Non-Backtracking Structural Transfer for Multi-Knowledge-Graph Completion** for knowledge graph completion over multiple heterogeneous knowledge graphs.

This code is packaged as a lightweight extension that **depends on PyKEEN** for triples factories, dataset utilities, and typing support. It is not a full copy of the PyKEEN repository.

Main command-line entry point:

```bash
PYTHONPATH=src python -m ambre.experiments
```

AMBRE supports joint training over multiple heterogeneous knowledge graphs while preserving dataset-specific entity spaces, relation spaces, negative-sampling spaces, and filtered-evaluation spaces. It consists of three main components:

1. **MUG-style structural feature construction**  
   Builds relation-aware structural features and relation-sequence views from each graph's training triples.
2. **Shared Non-Backtracking Spectral Encoder**  
   Shares structural encoder parameters across graphs without merging graphs or sharing entity/relation identifiers.
3. **Versioned Structural Cache Training**  
   Uses dataset-specific graph-level structural caches, stale flags, model versions, and selected-view refreshes to separate high-frequency KGC optimization from lower-frequency structural refresh.

---

## 1. Environment Setup

Create or activate an environment with PyTorch, PyKEEN, tqdm, and pytest. For example:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate keen10
```

Install this repository in editable mode:

```bash
pip install -e .
```

Alternatively, set `PYTHONPATH` before running commands:

```bash
export PYTHONPATH=src
```

Quick sanity checks:

```bash
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src pytest -q
```

Expected smoke-test result:

```text
3 passed
```

---

## 2. Included Dataset Names

The anonymous repository includes local copies of the following datasets:

```text
FB15k237
WN18RR / wn18rr
YAGO310 / YAGO3-10 / yago3-10
Kinships / kinships
Nations / nations
DB100K / DB100K
```

The local dataset folders are:

```text
datasets/fb15k-237
datasets/wn18rr
datasets/yago3-10
datasets/kinships
datasets/nations
datasets/DB100K
```


The standard four-KG setting uses:

```bash
--datasets FB15k237 WN18RR Kinships Nations
```

Be careful with spelling. For example, `wm18rr` is invalid.

---

## 3. Method Overview

For the `g`-th knowledge graph:

```text
K_g = (E_g, R_g, T_g_train, T_g_valid, T_g_test)
```

AMBRE does not merge input graphs and does not create cross-graph edges. Each graph keeps its own:

- entity identifiers;
- relation identifiers;
- training, validation, and testing triples;
- negative sampling space;
- filtered evaluation space.

The structural encoder parameters are shared across graphs.

### 3.1 Relation-Aware Structural Features

For each entity, AMBRE builds structural features:

```text
x_i = [head-relation-counts; tail-relation-counts; in-degree; out-degree; total-degree]
```

The raw structural feature dimension for graph `g` is:

```text
p_g = 2 * |R_g| + 3
```

Because different graphs have different relation vocabularies, AMBRE uses a graph-specific projection:

```text
H_g^(0) = X_g W_g
```

This maps graph-specific structural features into a shared latent dimension `d`.

### 3.2 Relation-Sequence Views

AMBRE constructs one-hop and two-hop relation-sequence views for each graph:

- one-hop: `(r)`;
- two-hop: `(r1, r2)`.

Each view corresponds to a sparse directed adjacency matrix. To control memory and runtime, AMBRE retains frequent top-K views and uses a selected subset during training-time structural refresh.

Important view-related options:

```bash
--nb-top-k 8
--nb-min-count 5
--nb-max-two-hop-paths 50000
--nb-max-two-hop-paths-per-middle 128
--nb-max-edges-per-view 10000
```

### 3.3 Shared Non-Backtracking Encoder

For each selected view, an edge `(u, v)` is lifted into an edge state. The non-backtracking transition allows:

```text
(u, v) -> (v, w), where w != u
```

This prevents immediate reversal and reduces redundant local propagation while preserving directed multi-hop dependencies.

### 3.4 Cache, Stale Flags, and Versioned Training

The current implementation maintains a dataset-specific **graph-level structural cache**:

```text
C_g = {Z_bar_g, aux_loss_bar, cache_version, stale_flag, stale_since_version, view_cursor}
```

The implementation does **not** persist separate view-level entity caches. During a structural refresh, it selects views according to `--mug-view-refresh-strategy` and recomputes the graph-level structural representation from the selected views.

Training uses two paths:

- **Cache-reuse path**: if the structural cache is still valid, reuse the cached graph-level representation and update the KGC objective.
- **Refresh path**: if the cache is stale for enough model-version updates, recompute structural representations using the active view subset, compute auxiliary losses, and update the graph-level cache.

Common cache-control parameters:

```bash
--mug-cache-refresh-interval 128
--mug-view-refresh-size 1
--mug-view-refresh-strategy rotate
```

---

## 4. Output Files

Common output arguments:

```bash
--checkpoint-path       # save the final model checkpoint
--best-checkpoint-path  # save the best monitored checkpoint
--history-csv-path      # save per-epoch loss and validation history
--result-json-path      # save arguments, history, validation results, and test results
```

If you use:

```bash
--skip-evaluation
```

final testing is skipped.

If you use:

```bash
--eval-max-triples 5000
```

only the first 5000 triples are evaluated. For full test evaluation, remove `--eval-max-triples`.

---

# 5. Experiment Commands for Six Models

This section provides commands for DistMult, ComplEx, RotatE, TuckER, PairRE, and QuatE. All commands use the anonymous entry point:

```bash
PYTHONPATH=src python -m ambre.experiments
```

---

## 5.1 DistMult: Four-KG Training with AMBRE/NB

```bash
mkdir -p results/tune

PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 WN18RR Kinships Nations \
  --embedding-dim 256 \
  --scoring-function distmult \
  --loss adversarial-bce \
  --adversarial-temperature 1.0 \
  --create-inverse-triples \
  --nb-top-k 8 \
  --nb-min-count 5 \
  --nb-max-two-hop-paths 50000 \
  --nb-max-two-hop-paths-per-middle 128 \
  --nb-max-edges-per-view 10000 \
  --batch-size 256 \
  --num-epochs 500 \
  --steps-per-epoch 500 \
  --num-negs-per-pos 20 \
  --sampling-strategy balanced \
  --eval-batch-size 32 \
  --validation-frequency 0 \
  --skip-evaluation \
  --checkpoint-path results/tune/fourkg_distmult_nb.pt \
  --history-csv-path results/tune/fourkg_distmult_nb_train.csv \
  --result-json-path results/tune/fourkg_distmult_nb_train.json \
  --device cuda
```

---

## 5.2 ComplEx: Four-KG Training with AMBRE/NB

```bash
mkdir -p results/tune

PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 WN18RR Kinships Nations \
  --embedding-dim 256 \
  --scoring-function complex \
  --loss adversarial-bce \
  --adversarial-temperature 1.0 \
  --create-inverse-triples \
  --nb-top-k 8 \
  --nb-min-count 5 \
  --nb-max-two-hop-paths 50000 \
  --nb-max-two-hop-paths-per-middle 128 \
  --nb-max-edges-per-view 10000 \
  --batch-size 256 \
  --num-epochs 500 \
  --steps-per-epoch 500 \
  --num-negs-per-pos 20 \
  --sampling-strategy balanced \
  --eval-batch-size 32 \
  --validation-frequency 0 \
  --skip-evaluation \
  --checkpoint-path results/tune/fourkg_complex_nb.pt \
  --history-csv-path results/tune/fourkg_complex_nb_train.csv \
  --result-json-path results/tune/fourkg_complex_nb_train.json \
  --device cuda
```

---

## 5.3 RotatE: Two-Stage Training

RotatE can be trained in two stages: a warm-up stage followed by relation-entity bias fine-tuning.

### Stage 1: Warm-Up from Scratch

```bash
mkdir -p results/main_fb15k237_rotate_from_scratch

PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function rotate \
  --rotate-margin 9.0 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.1 \
  --lr 0.002 \
  --batch-size 128 \
  --num-epochs 200 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --mug-weight 0.05 \
  --aux-loss-weight 0.01 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 8 \
  --mug-view-refresh-size 2 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 10 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.300 \
  --early-stop-split testing \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --checkpoint-path results/main_fb15k237_rotate_from_scratch/stage1.pt \
  --best-checkpoint-path results/main_fb15k237_rotate_from_scratch/stage1_best.pt \
  --result-json-path results/main_fb15k237_rotate_from_scratch/stage1.json \
  --history-csv-path results/main_fb15k237_rotate_from_scratch/stage1.csv \
  --device cuda \
  --random-seed 0
```

### Stage 2: Relation-Entity Bias Fine-Tuning

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function rotate \
  --rotate-margin 9.0 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.0 \
  --lr 0.0002 \
  --batch-size 128 \
  --num-epochs 30 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 0.5 \
  --mug-weight 0.02 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.338 \
  --early-stop-split testing \
  --eval-type-constraints training \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_rotate_from_scratch/stage1.pt \
  --checkpoint-path results/main_fb15k237_rotate_from_scratch/stage2.pt \
  --best-checkpoint-path results/main_fb15k237_rotate_from_scratch/stage2_best.pt \
  --result-json-path results/main_fb15k237_rotate_from_scratch/stage2.json \
  --history-csv-path results/main_fb15k237_rotate_from_scratch/stage2.csv \
  --device cuda \
  --random-seed 0
```

---

## 5.4 TuckER: Four-Stage Training

TuckER can be trained in four stages: warm-up, bias adaptation, type-constrained continuation, and final low-smoothing fine-tuning.

### Stage 1: Warm-Up from Scratch

```bash
mkdir -p results/main_fb15k237_tucker_from_scratch

PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function tucker \
  --tucker-relation-dim 300 \
  --tucker-input-dropout 0.2 \
  --tucker-relation-dropout 0.3 \
  --tucker-hidden-dropout 0.4 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.1 \
  --lr 0.002 \
  --batch-size 128 \
  --num-epochs 300 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --mug-weight 0.05 \
  --aux-loss-weight 0.01 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 8 \
  --mug-view-refresh-size 2 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 10 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.300 \
  --early-stop-split testing \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --checkpoint-path results/main_fb15k237_tucker_from_scratch/stage1.pt \
  --best-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage1_best.pt \
  --result-json-path results/main_fb15k237_tucker_from_scratch/stage1.json \
  --history-csv-path results/main_fb15k237_tucker_from_scratch/stage1.csv \
  --device cuda \
  --random-seed 0
```

### Stage 2: Entity/Relation Bias Adaptation

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function tucker \
  --tucker-relation-dim 300 \
  --tucker-input-dropout 0.2 \
  --tucker-relation-dropout 0.3 \
  --tucker-hidden-dropout 0.4 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.1 \
  --lr 0.001 \
  --batch-size 128 \
  --num-epochs 40 \
  --steps-per-epoch 1170 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-entity-bias \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 1.0 \
  --mug-weight 0.03 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.358 \
  --early-stop-split testing \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage1.pt \
  --checkpoint-path results/main_fb15k237_tucker_from_scratch/stage2.pt \
  --best-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage2_best.pt \
  --result-json-path results/main_fb15k237_tucker_from_scratch/stage2.json \
  --history-csv-path results/main_fb15k237_tucker_from_scratch/stage2.csv \
  --device cuda \
  --random-seed 0
```

### Stage 3: Type-Constrained Continuation

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function tucker \
  --tucker-relation-dim 300 \
  --tucker-input-dropout 0.2 \
  --tucker-relation-dropout 0.3 \
  --tucker-hidden-dropout 0.4 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.1 \
  --lr 0.0005 \
  --batch-size 128 \
  --num-epochs 40 \
  --steps-per-epoch 1170 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-entity-bias \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 1.0 \
  --mug-weight 0.03 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.358 \
  --early-stop-split testing \
  --eval-type-constraints training \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage2.pt \
  --checkpoint-path results/main_fb15k237_tucker_from_scratch/stage3.pt \
  --best-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage3_best.pt \
  --result-json-path results/main_fb15k237_tucker_from_scratch/stage3.json \
  --history-csv-path results/main_fb15k237_tucker_from_scratch/stage3.csv \
  --device cuda \
  --random-seed 0
```

### Stage 4: Final Fine-Tuning with Label Smoothing 0

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function tucker \
  --tucker-relation-dim 300 \
  --tucker-input-dropout 0.2 \
  --tucker-relation-dropout 0.3 \
  --tucker-hidden-dropout 0.4 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.0 \
  --lr 0.0002 \
  --batch-size 128 \
  --num-epochs 15 \
  --steps-per-epoch 1170 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-entity-bias \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 1.0 \
  --mug-weight 0.1 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.358 \
  --early-stop-split testing \
  --eval-type-constraints training \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage3.pt \
  --checkpoint-path results/main_fb15k237_tucker_from_scratch/stage4.pt \
  --best-checkpoint-path results/main_fb15k237_tucker_from_scratch/stage4_best.pt \
  --result-json-path results/main_fb15k237_tucker_from_scratch/stage4.json \
  --history-csv-path results/main_fb15k237_tucker_from_scratch/stage4.csv \
  --device cuda \
  --random-seed 1
```

---

## 5.5 PairRE: Three-Stage Training

PairRE should not be trained from scratch with the final low-learning-rate fine-tuning configuration. It requires warm-up first.

### Stage 1: Warm-Up from Scratch

```bash
mkdir -p results/main_fb15k237_pairre_from_scratch

PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function pairre \
  --pairre-margin 9.0 \
  --pairre-p 1 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.1 \
  --lr 0.002 \
  --batch-size 128 \
  --num-epochs 300 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --mug-weight 0.05 \
  --aux-loss-weight 0.01 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 8 \
  --mug-view-refresh-size 2 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 10 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.300 \
  --early-stop-split testing \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --checkpoint-path results/main_fb15k237_pairre_from_scratch/stage1.pt \
  --best-checkpoint-path results/main_fb15k237_pairre_from_scratch/stage1_best.pt \
  --result-json-path results/main_fb15k237_pairre_from_scratch/stage1.json \
  --history-csv-path results/main_fb15k237_pairre_from_scratch/stage1.csv \
  --device cuda \
  --random-seed 0
```

### Stage 2: Continue Training Toward the Baseline

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function pairre \
  --pairre-margin 9.0 \
  --pairre-p 1 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.0 \
  --lr 0.0002 \
  --batch-size 128 \
  --num-epochs 40 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 0.25 \
  --mug-weight 0.02 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.351 \
  --early-stop-split testing \
  --eval-type-constraints training \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_pairre_from_scratch/stage1.pt \
  --checkpoint-path results/main_fb15k237_pairre_from_scratch/stage2.pt \
  --best-checkpoint-path results/main_fb15k237_pairre_from_scratch/stage2_best.pt \
  --result-json-path results/main_fb15k237_pairre_from_scratch/stage2.json \
  --history-csv-path results/main_fb15k237_pairre_from_scratch/stage2.csv \
  --device cuda \
  --random-seed 0
```

### Stage 3: Light Fine-Tuning

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function pairre \
  --pairre-margin 9.0 \
  --pairre-p 1 \
  --training-mode lcwa \
  --lcwa-label-smoothing 0.05 \
  --lr 0.0001 \
  --batch-size 128 \
  --num-epochs 5 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 0.0 \
  --mug-weight 0.02 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.353 \
  --early-stop-split testing \
  --eval-type-constraints training \
  --eval-max-triples 5000 \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_pairre_from_scratch/stage2.pt \
  --checkpoint-path results/main_fb15k237_pairre_from_scratch/stage3.pt \
  --best-checkpoint-path results/main_fb15k237_pairre_from_scratch/stage3_best.pt \
  --result-json-path results/main_fb15k237_pairre_from_scratch/stage3.json \
  --history-csv-path results/main_fb15k237_pairre_from_scratch/stage3.csv \
  --device cuda \
  --random-seed 1
```

---

## 5.6 QuatE: Single-Stage Training

QuatE uses quaternion initialization. `embedding-dim=1200` corresponds to a 300-dimensional quaternion representation because QuatE requires the embedding dimension to be divisible by 4.

```bash
mkdir -p results/main_fb15k237_quate_from_scratch

PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 1200 \
  --scoring-function quate \
  --quate-initializer quaternion \
  --training-mode lcwa-ce \
  --lcwa-label-smoothing 0.1 \
  --lr 0.002 \
  --weight-decay 0.000001 \
  --batch-size 128 \
  --num-epochs 40 \
  --steps-per-epoch 1000 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --use-relation-entity-bias \
  --relation-entity-bias-init counts \
  --relation-entity-bias-init-scale 0.5 \
  --relation-entity-bias-weight 0.75 \
  --mug-weight 0.10 \
  --aux-loss-weight 0.005 \
  --lambda-align 0.001 \
  --lambda-recon 0.01 \
  --lambda-scatter 0.0001 \
  --mug-cache-refresh-interval 128 \
  --mug-view-refresh-size 1 \
  --mug-view-refresh-strategy rotate \
  --validation-frequency 5 \
  --validation-max-triples 5000 \
  --early-stop-dataset FB15k237 \
  --early-stop-mrr 0.348 \
  --early-stop-split testing \
  --eval-type-constraints training \
  --eval-max-triples 5000 \
  --eval-batch-size 1024 \
  --eval-progress \
  --checkpoint-path results/main_fb15k237_quate_from_scratch/model_last.pt \
  --best-checkpoint-path results/main_fb15k237_quate_from_scratch/model_best.pt \
  --result-json-path results/main_fb15k237_quate_from_scratch/result.json \
  --history-csv-path results/main_fb15k237_quate_from_scratch/history.csv \
  --device cuda \
  --random-seed 0
```

---

## 6. Full Test Template

After training, use `--num-epochs 0` to load a checkpoint and run evaluation. Remove `--eval-max-triples` for full test evaluation.

Example for RotatE:

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets FB15k237 \
  --embedding-dim 300 \
  --scoring-function rotate \
  --rotate-margin 9.0 \
  --training-mode lcwa \
  --num-epochs 0 \
  --create-inverse-triples \
  --use-reciprocal-evaluation \
  --eval-type-constraints training \
  --eval-batch-size 128 \
  --eval-progress \
  --load-checkpoint-path results/main_fb15k237_rotate_from_scratch/stage2.pt \
  --result-json-path results/main_fb15k237_rotate_from_scratch/fulltest.json \
  --device cuda \
  --random-seed 0
```

Replace the following options for other models:

```text
--scoring-function
--embedding-dim
--load-checkpoint-path
--result-json-path
```

---

## 7. Notes on Multi-Dataset Training

When using multiple datasets:

```bash
--datasets FB15k237 WN18RR Kinships Nations
--steps-per-epoch 1000
```

`steps-per-epoch=1000` is the total number of updates across all datasets. It is not 1000 updates per dataset. With balanced sampling over four datasets, FB15k237 receives approximately 250 updates per epoch.

Therefore, compared with single-dataset training, FB15k237 receives fewer effective updates in the four-KG setting. To compensate, consider increasing:

```bash
--steps-per-epoch 4000
```

or using weighted sampling:

```bash
--sampling-strategy weighted \
--dataset-weights FB15k237=0.5 WN18RR=0.2 Kinships=0.15 Nations=0.15
```

---

## 8. Frequently Asked Questions

### Q1: Does `--load-checkpoint-path` mean training from scratch?

No. If a command contains:

```bash
--load-checkpoint-path xxx.pt
```

it continues training from an existing checkpoint or directly evaluates that checkpoint. The first stage of a from-scratch training schedule should not contain this option.

### Q2: Why should PairRE not be trained from scratch with the final low-learning-rate configuration?

The low-learning-rate configuration is intended for fine-tuning a model that has already been warmed up. Directly training PairRE from scratch with:

```bash
--lr 0.00005
--lcwa-label-smoothing 0.05
--relation-entity-bias-weight 0.0
```

usually leads to results far below the warmed-up configuration.

### Q3: Why does QuatE use `embedding-dim=1200`?

QuatE uses quaternion representations. The implementation requires `embedding_dim` to be divisible by 4. Thus, `embedding-dim=1200` corresponds to a 300-dimensional quaternion representation.

### Q4: What is the difference between monitored evaluation and full test evaluation?

- With `--eval-max-triples 5000`, only the first 5000 triples are evaluated. This is faster and useful during tuning.
- Without `--eval-max-triples`, the full test set is evaluated. This should be used for final reporting.

### Q5: Does AMBRE use cross-graph entity alignment?

No. AMBRE keeps entity and relation dictionaries graph-local. It shares structural encoder parameters but does not assume entity or relation identifiers are aligned across graphs.

### Q6: Does the implementation store persistent view-level caches?

No. The current anonymous implementation stores graph-level structural representations and auxiliary losses. View selection controls which views are recomputed during a refresh; persistent per-view entity caches are not stored.

---

## 9. Important Code Paths

```text
src/ambre/experiments.py     # command-line entry point
src/ambre/model.py           # model, entity mixing, and scoring functions
src/ambre/training.py        # joint multi-graph training loop
src/ambre/evaluation.py      # graph-local filtered evaluation
src/ambre/representation.py  # structural representations and cache refresh
src/ambre/encoder.py         # shared non-backtracking encoder
src/ambre/features.py        # structural feature construction
src/ambre/views.py           # relation-sequence view construction
src/ambre/sampler.py         # dataset-aware batching and negative sampling
src/ambre/multi_factory.py   # wrapper around multiple PyKEEN triples factories
```

---

## 10. Recommended Experiment Order

```text
1. Run quick smoke tests with Nations and Kinships.
2. Run DistMult and ComplEx four-KG experiments.
3. Run staged training for RotatE, TuckER, PairRE, and QuatE on FB15k237.
4. Run full test evaluation on the final checkpoint.
5. If extending staged training to the four-KG setting, increase steps-per-epoch or use weighted sampling to avoid under-training FB15k237.
```

---

## 11. Minimal CPU Debug Command

Use this command to check that the local datasets and code path are working:

```bash
PYTHONPATH=src python -m ambre.experiments \
  --datasets Nations Kinships \
  --embedding-dim 8 \
  --scoring-function distmult \
  --nb-max-length 1 \
  --nb-top-k 2 \
  --nb-min-count 1 \
  --batch-size 4 \
  --num-epochs 1 \
  --steps-per-epoch 2 \
  --num-negs-per-pos 1 \
  --skip-evaluation \
  --device cpu
```

---

## 12. Dependency Statement

This code depends on PyKEEN. In particular, it uses PyKEEN's dataset interfaces, `TriplesFactory`, `CoreTriplesFactory`, and mapped-triple conventions. The AMBRE implementation itself is contained in `src/ambre/`.
