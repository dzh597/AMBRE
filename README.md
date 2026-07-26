# AMBRE Anonymous Implementation

This repository contains a compact anonymous implementation of **AMBRE** for knowledge graph completion over multiple heterogeneous knowledge graphs.

AMBRE preserves graph boundaries: each dataset keeps its own entity IDs, relation IDs, negative-sampling space, and filtered-evaluation space. Structural parameters are shared across graphs through relation-incidence features, relation-sequence views, and a shared non-backtracking encoder.

## Code layout

```text
src/ambre/
  features.py        structural entity features
  views.py           one-hop and two-hop relation-sequence views
  encoder.py         shared non-backtracking encoder
  representation.py  structural representation cache
  model.py           KGE scoring model and AMBRE integration
  sampler.py         graph-aware batch and negative sampling
  training.py        joint multi-graph training loop
  evaluation.py      graph-local filtered evaluation
  experiments.py     command-line runner
tests/
  test_smoke.py      minimal import/forward/cache tests
```

## Installation

This code depends on **PyKEEN**. It uses PyKEEN datasets, triples factories, and typing utilities.

Install in editable mode:

```bash
pip install -e .
```

For tests:

```bash
pip install -e '.[test]'
pytest -q
```

If you do not install the package, set `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

## Quick checks

```bash
python -m compileall -q src
pytest -q
```

## Included datasets

The repository includes local copies of the benchmark datasets used by the runner:

- `FB15k237` (`datasets/fb15k-237`)
- `WN18RR` (`datasets/wn18rr`)
- `YAGO310` / `YAGO3-10` (`datasets/yago3-10`)
- `Nations` (`datasets/nations`)
- `Kinships` (`datasets/kinships`)
- `UMLS` (`datasets/umls`)

Wikidata5M is intentionally not included. When one of the above dataset names is used, the runner first loads the local files from `datasets/`; PyKEEN remains required for `TriplesFactory` and dataset utilities.

## Minimal training command

The command-line entry point is:

```bash
python -m ambre.experiments
```

A small local-dataset example:

```bash
python -m ambre.experiments \
  --datasets Nations Kinships \
  --embedding-dim 32 \
  --scoring-function distmult \
  --nb-max-length 1 \
  --nb-top-k 2 \
  --nb-min-count 1 \
  --batch-size 16 \
  --num-epochs 1 \
  --steps-per-epoch 2 \
  --num-negs-per-pos 1 \
  --skip-evaluation \
  --device cpu
```

A larger local-dataset example:

```bash
python -m ambre.experiments \
  --datasets FB15k237 WN18RR \
  --embedding-dim 128 \
  --scoring-function distmult \
  --nb-top-k 8 \
  --nb-min-count 5 \
  --batch-size 128 \
  --num-epochs 1 \
  --steps-per-epoch 10 \
  --num-negs-per-pos 4 \
  --skip-evaluation \
  --device cpu
```

## Common options

```text
--datasets                         dataset names, e.g. FB15k237 WN18RR YAGO310 Nations Kinships UMLS
--embedding-dim                    entity/relation embedding dimension
--scoring-function                 distmult, complex, rotate, tucker, quate, pairre, affine, etc.
--nb-top-k                         number of retained relation-sequence views
--nb-max-length                    maximum relation-sequence length, currently 1 or 2
--mug-cache-refresh-interval       number of parameter-version updates before structural refresh
--mug-view-refresh-size            number of views refreshed during training
--mug-view-refresh-strategy        rotate, sample, or all
--skip-evaluation                  skip final filtered evaluation
```

## Dependency note

This is not a fork of the full PyKEEN repository. It is a small extension package that **requires PyKEEN** to be installed.
