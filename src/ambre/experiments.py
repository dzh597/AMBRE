"""Command-line runner for the AMBRE MVP."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from pykeen.datasets import FB15k237, Kinships, Nations, UMLS, WN18RR, YAGO310
from pykeen.triples import TriplesFactory

from .evaluation import evaluate_all
from .model import MUGKGC
from .multi_factory import multi_factory_from_datasets
from .training import train_joint


DATASET_CLASSES = {
    "Nations": Nations,
    "nations": Nations,
    "Kinships": Kinships,
    "kinships": Kinships,
    "UMLS": UMLS,
    "umls": UMLS,
    "FB15k237": FB15k237,
    "fb15k237": FB15k237,
    "WN18RR": WN18RR,
    "wn18rr": WN18RR,
    "YAGO310": YAGO310,
    "YAGO3-10": YAGO310,
    "yago3-10": YAGO310,
}

LOCAL_DATASET_PATHS = {
    "Nations": Path(__file__).resolve().parents[2].joinpath("datasets", "nations"),
    "nations": Path(__file__).resolve().parents[2].joinpath("datasets", "nations"),
    "Kinships": Path(__file__).resolve().parents[2].joinpath("datasets", "kinships"),
    "kinships": Path(__file__).resolve().parents[2].joinpath("datasets", "kinships"),
    "UMLS": Path(__file__).resolve().parents[2].joinpath("datasets", "umls"),
    "umls": Path(__file__).resolve().parents[2].joinpath("datasets", "umls"),
    "FB15k237": Path(__file__).resolve().parents[2].joinpath("datasets", "fb15k-237"),
    "fb15k237": Path(__file__).resolve().parents[2].joinpath("datasets", "fb15k-237"),
    "WN18RR": Path(__file__).resolve().parents[2].joinpath("datasets", "wn18rr"),
    "wn18rr": Path(__file__).resolve().parents[2].joinpath("datasets", "wn18rr"),
    "YAGO310": Path(__file__).resolve().parents[2].joinpath("datasets", "yago3-10"),
    "YAGO3-10": Path(__file__).resolve().parents[2].joinpath("datasets", "yago3-10"),
    "yago3-10": Path(__file__).resolve().parents[2].joinpath("datasets", "yago3-10"),
}


@dataclass
class LocalPathDataset:
    """A tiny dataset container matching the attributes used by the runner."""

    training: TriplesFactory
    validation: TriplesFactory
    testing: TriplesFactory


def _read_dict(path: Path) -> dict[str, int]:
    """Read OpenKE-style ``id<TAB>label`` mapping files."""
    mapping: dict[str, int] = {}
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            idx, label = line.split("\t", maxsplit=1)
            mapping[label] = int(idx)
    return mapping


def _load_local_path_dataset(path: Path, *, create_inverse_triples: bool = False) -> LocalPathDataset:
    """Load train/valid/test triples from an existing local directory."""
    entity_path = path.joinpath("entities.dict")
    relation_path = path.joinpath("relations.dict")
    has_dicts = entity_path.is_file() and relation_path.is_file()
    entity_to_id = _read_dict(entity_path) if has_dicts else None
    relation_to_id = _read_dict(relation_path) if has_dicts else None

    def _factory(split: str, *, inverse: bool = False) -> TriplesFactory:
        return TriplesFactory.from_path(
            path.joinpath(f"{split}.txt"),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            compact_id=False,
            create_inverse_triples=inverse,
        )

    if not has_dicts:
        training = TriplesFactory.from_path(
            path.joinpath("train.txt"),
            create_inverse_triples=create_inverse_triples,
        )
        return LocalPathDataset(
            training=training,
            validation=TriplesFactory.from_path(
                path.joinpath("valid.txt"),
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id,
                compact_id=False,
            ),
            testing=TriplesFactory.from_path(
                path.joinpath("test.txt"),
                entity_to_id=training.entity_to_id,
                relation_to_id=training.relation_to_id,
                compact_id=False,
            ),
        )

    return LocalPathDataset(
        training=_factory("train", inverse=create_inverse_triples),
        validation=_factory("valid"),
        testing=_factory("test"),
    )


def _augment_inverse_training_triples(dataset: object) -> None:
    """Materialize inverse triples for the custom training loop.

    PyKEEN's ``create_inverse_triples=True`` doubles ``num_relations`` but keeps
    ``mapped_triples`` in its original form because its own training loops create
    inverse scores internally. This custom loop samples from ``mapped_triples``
    directly, so we explicitly append ``(t, r + offset, h)`` for training only.
    """
    training = dataset.training
    if not getattr(training, "create_inverse_triples", False):
        return
    mapped_triples = training.mapped_triples
    relation_offset = training.num_relations // 2
    if mapped_triples.numel() == 0 or int(mapped_triples[:, 1].max()) >= relation_offset:
        return
    inverse = mapped_triples[:, [2, 1, 0]].clone()
    inverse[:, 1] += relation_offset
    training.mapped_triples = torch.cat([mapped_triples, inverse], dim=0)
    # The inverse triples have been materialized. Keep relation cardinality
    # doubled, but avoid accidental second inverse handling downstream.
    training.create_inverse_triples = False


def _load_datasets(names: list[str], *, create_inverse_triples: bool = False) -> dict[str, object]:
    unknown = sorted(set(names).difference(DATASET_CLASSES))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Available: {sorted(DATASET_CLASSES)}")
    datasets = {}
    for name in names:
        local_path = LOCAL_DATASET_PATHS.get(name)
        if local_path is not None and local_path.is_dir():
            dataset = _load_local_path_dataset(local_path, create_inverse_triples=create_inverse_triples)
        else:
            dataset_cls = DATASET_CLASSES[name]
            if dataset_cls is None:
                raise ValueError(f"Local dataset files for {name!r} were not found at {local_path}.")
            dataset = dataset_cls(create_inverse_triples=create_inverse_triples)
        _augment_inverse_training_triples(dataset)
        datasets[name] = dataset
    return datasets


def _flatten_metrics(prefix: str, results: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{prefix}.{dataset_name}.{metric_name}": value
        for dataset_name, metrics in results.items()
        for metric_name, value in metrics.items()
    }


def _print_results(title: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}:")
    for name, metrics in results.items():
        print(f"{name}: " + " ".join(f"{metric}={value:.4f}" for metric, value in metrics.items()))


def _save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _save_history_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    path: str | Path,
    *,
    model: MUGKGC,
    args: argparse.Namespace,
    dataset_names: list[str],
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint with optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "dataset_names": dataset_names,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _parse_dataset_weights(values: list[str] | None) -> dict[str, float] | None:
    """Parse CLI values like ``FB15k237=0.5 WN18RR=0.35``."""
    if not values:
        return None
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid dataset weight {value!r}; expected NAME=WEIGHT.")
        name, weight = value.split("=", maxsplit=1)
        result[name] = float(weight)
    return result



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMBRE joint multi-KG training MVP with shared non-backtracking encoding")
    parser.add_argument("--datasets", nargs="+", default=["Nations", "Kinships"])
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--feature-mode", default="relation_incidence")
    parser.add_argument("--nb-max-length", dest="nb_max_length", type=int, default=2)
    parser.add_argument("--nb-top-k", dest="nb_top_k", type=int, default=16)
    parser.add_argument("--nb-min-count", dest="nb_min_count", type=int, default=10)
    parser.add_argument("--nb-max-two-hop-paths", dest="nb_max_two_hop_paths", type=int, default=200_000)
    parser.add_argument("--nb-max-two-hop-paths-per-middle", dest="nb_max_two_hop_paths_per_middle", type=int, default=512)
    parser.add_argument("--nb-max-edges-per-view", dest="nb_max_edges_per_view", type=int, default=50_000)
    parser.add_argument("--lambda-align", type=float, default=0.01)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--lambda-scatter", type=float, default=0.001)
    parser.add_argument("--entity-embedding-weight", type=float, default=1.0)
    parser.add_argument("--mug-weight", type=float, default=1.0)
    parser.add_argument("--aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--disable-mug", action="store_true")
    parser.add_argument("--max-structural-entities", type=int, default=1_000_000)
    parser.add_argument(
        "--scoring-function",
        choices=[
            "distmult",
            "affine",
            "complex",
            "tucker",
            "rotate",
            "quate",
            "pairre",
            "affine_distmult",
            "affine_complex",
        ],
        default="distmult",
    )
    parser.add_argument("--affine-p", type=int, default=2)
    parser.add_argument("--affine-aggr", choices=["norm", "pow"], default="norm")
    parser.add_argument("--affine-init-margin", type=float, default=3.0)
    parser.add_argument("--rotate-margin", type=float, default=9.0)
    parser.add_argument("--tucker-relation-dim", type=int)
    parser.add_argument("--tucker-input-dropout", type=float, default=0.0)
    parser.add_argument("--tucker-relation-dropout", type=float, default=0.0)
    parser.add_argument("--tucker-hidden-dropout", type=float, default=0.0)
    parser.add_argument("--no-tucker-batch-norm", action="store_true")
    parser.add_argument("--pairre-margin", type=float, default=9.0)
    parser.add_argument("--pairre-p", type=int, choices=[1, 2], default=1)
    parser.add_argument("--hybrid-weight", type=float, default=1.0)
    parser.add_argument(
        "--quate-affine-weight",
        type=float,
        default=0.0,
        help="Optional affine residual weight for scoring_function='quate'.",
    )
    parser.add_argument(
        "--quate-scale-weight",
        type=float,
        default=0.0,
        help="Optional relation-wise quaternion scale residual weight for scoring_function='quate'.",
    )
    parser.add_argument(
        "--quate-initializer",
        choices=["uniform", "quaternion"],
        default="uniform",
        help="initializer for scoring_function='quate'; 'quaternion' matches PyKEEN's QuatE-style modulus/phase init",
    )
    parser.add_argument("--use-entity-bias", action="store_true")
    parser.add_argument("--use-relation-entity-bias", action="store_true")
    parser.add_argument("--relation-entity-bias-init", choices=["zeros", "counts"], default="zeros")
    parser.add_argument("--relation-entity-bias-init-scale", type=float, default=0.1)
    parser.add_argument(
        "--relation-entity-bias-weight",
        type=float,
        default=1.0,
        help="runtime multiplier for relation/entity bias scores",
    )
    parser.add_argument("--no-dimension-alignment", action="store_true")
    parser.add_argument("--no-nb-encoder", dest="no_nb_encoder", action="store_true", help="disable the shared non-backtracking encoder")
    parser.add_argument("--no-alignment-loss", action="store_true")
    parser.add_argument("--no-reconstruction-loss", action="store_true")
    parser.add_argument("--no-scatter-loss", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--mug-cache-refresh-interval",
        type=int,
        default=4,
        help="reuse cached structural outputs for this many parameter updates before refreshing",
    )
    parser.add_argument("--mug-view-refresh-size", type=int, default=8, help="number of views to use per structural refresh")
    parser.add_argument(
        "--mug-view-refresh-strategy",
        choices=["all", "rotate", "sample"],
        default="rotate",
        help="how to choose views during training-time structural refreshes",
    )
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-negs-per-pos", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=["softplus", "adversarial-bce", "nssa"], default="softplus")
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=9.0)
    parser.add_argument("--training-mode", choices=["slcwa", "lcwa", "lcwa-ce"], default="slcwa")
    parser.add_argument(
        "--lcwa-loss",
        choices=["bce", "ce", "mce"],
        default="bce",
        help=(
            "Loss used with --training-mode lcwa. "
            "'mce' is a multi-positive cross-entropy over all true tails for each (h,r)."
        ),
    )
    parser.add_argument("--lcwa-label-smoothing", type=float, default=0.0)
    parser.add_argument("--sampling-strategy", choices=["balanced", "proportional", "temperature", "weighted"], default="balanced")
    parser.add_argument("--dataset-sampling-temperature", type=float, default=0.5)
    parser.add_argument("--dataset-weights", nargs="*")
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-max-triples", type=int)
    parser.add_argument("--eval-progress", action="store_true")
    parser.add_argument("--use-reciprocal-evaluation", action="store_true")
    parser.add_argument(
        "--eval-type-constraints",
        choices=["none", "training", "training_validation"],
        default="none",
    )
    parser.add_argument("--validation-frequency", type=int, default=0)
    parser.add_argument("--validation-max-triples", type=int)
    parser.add_argument("--early-stop-dataset", help="stop when this dataset reaches --early-stop-mrr on the monitored split")
    parser.add_argument("--early-stop-mrr", type=float, help="target MRR for early stopping")
    parser.add_argument(
        "--early-stop-split",
        choices=["validation", "testing"],
        default="validation",
        help="split monitored by --early-stop-mrr",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--result-json-path")
    parser.add_argument("--history-csv-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--best-checkpoint-path",
        help=(
            "save a checkpoint whenever the monitored validation/testing MRR improves; "
            "the monitored split is controlled by --early-stop-split when --early-stop-mrr "
            "is set, otherwise validation is used"
        ),
    )
    parser.add_argument("--load-checkpoint-path")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument(
        "--keep-initialized-relation-entity-bias-on-load",
        action="store_true",
        help=(
            "when loading a checkpoint, keep relation/entity bias parameters initialized "
            "from the current run instead of overwriting them from the checkpoint"
        ),
    )
    parser.add_argument("--create-inverse-triples", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    datasets = _load_datasets(args.datasets, create_inverse_triples=args.create_inverse_triples)
    multi_factory = multi_factory_from_datasets(datasets)
    print("Loaded datasets:")
    for name in multi_factory.get_dataset_names():
        training = multi_factory.get_training_factory(name)
        testing = multi_factory.get_testing_factory(name)
        print(
            f"  {name}: entities={training.num_entities} relations={training.num_relations} "
            f"train={training.mapped_triples.shape[0]} test={testing.mapped_triples.shape[0] if testing else 0}"
        )

    model = MUGKGC(
        multi_factory=multi_factory,
        embedding_dim=args.embedding_dim,
        feature_mode=args.feature_mode,
        nb_max_length=args.nb_max_length,
        nb_top_k=args.nb_top_k,
        nb_min_count=args.nb_min_count,
        nb_max_two_hop_paths=args.nb_max_two_hop_paths,
        nb_max_two_hop_paths_per_middle=args.nb_max_two_hop_paths_per_middle,
        nb_max_edges_per_view=args.nb_max_edges_per_view,
        lambda_align=0.0 if args.no_alignment_loss else args.lambda_align,
        lambda_recon=0.0 if args.no_reconstruction_loss else args.lambda_recon,
        lambda_scatter=0.0 if args.no_scatter_loss else args.lambda_scatter,
        dropout=args.dropout,
        use_dimension_alignment=not args.no_dimension_alignment,
        use_nb_encoder=not args.no_nb_encoder,
        cache_refresh_interval=args.mug_cache_refresh_interval,
        view_refresh_size=args.mug_view_refresh_size,
        view_refresh_strategy=args.mug_view_refresh_strategy,
        entity_embedding_weight=args.entity_embedding_weight,
        mug_weight=0.0 if args.disable_mug else args.mug_weight,
        max_structural_entities=0 if args.disable_mug else args.max_structural_entities,
        scoring_function=args.scoring_function,
        affine_p=args.affine_p,
        affine_aggr=args.affine_aggr,
        affine_init_margin=args.affine_init_margin,
        rotate_margin=args.rotate_margin,
        tucker_relation_dim=args.tucker_relation_dim,
        tucker_input_dropout=args.tucker_input_dropout,
        tucker_relation_dropout=args.tucker_relation_dropout,
        tucker_hidden_dropout=args.tucker_hidden_dropout,
        tucker_batch_norm=not args.no_tucker_batch_norm,
        pairre_margin=args.pairre_margin,
        pairre_p=args.pairre_p,
        hybrid_weight=args.hybrid_weight,
        quate_affine_weight=args.quate_affine_weight,
        quate_scale_weight=args.quate_scale_weight,
        quate_initializer=args.quate_initializer,
        use_entity_bias=args.use_entity_bias,
        use_relation_entity_bias=args.use_relation_entity_bias,
        relation_entity_bias_init=args.relation_entity_bias_init,
        relation_entity_bias_init_scale=args.relation_entity_bias_init_scale,
        relation_entity_bias_weight=args.relation_entity_bias_weight,
    ).to(args.device)
    skipped_structural = sorted(set(multi_factory.get_dataset_names()).difference(model.structural_dataset_names))
    if args.disable_mug:
        print("AMBRE structural features disabled (--disable-mug).")
    elif skipped_structural:
        print(
            "Skipping structural AMBRE features for large dataset(s): "
            + ", ".join(skipped_structural)
            + f" (max_structural_entities={args.max_structural_entities})"
        )
    if args.load_checkpoint_path:
        checkpoint_path = Path(args.load_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        checkpoint_dataset_names = checkpoint.get("dataset_names")
        current_dataset_names = multi_factory.get_dataset_names()
        if checkpoint_dataset_names is not None and list(checkpoint_dataset_names) != list(current_dataset_names):
            if not args.allow_partial_checkpoint:
                raise ValueError(
                    "Checkpoint dataset order does not match current datasets: "
                    f"checkpoint={checkpoint_dataset_names}, current={current_dataset_names}. "
                    "Pass --allow-partial-checkpoint to load matching dataset modules only."
                )
            print(
                "Partially loading checkpoint dataset modules: "
                f"checkpoint={checkpoint_dataset_names}, current={current_dataset_names}"
            )
        checkpoint_state = checkpoint["model_state_dict"]
        if args.keep_initialized_relation_entity_bias_on_load:
            checkpoint_state = {
                key: value
                for key, value in checkpoint_state.items()
                if not key.startswith("relation_head_entity_biases.")
                and not key.startswith("relation_tail_entity_biases.")
            }
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        relevant_missing = [
            key
            for key in missing_keys
            if not key.startswith("affine_relation_")
            and not key.startswith("tucker_cores.")
            and not key.startswith("tucker_bn")
            and not key.startswith("pairre_relation_tail_embeddings.")
            and not key.startswith("quate_relation_scales.")
            and not key.startswith("head_entity_biases.")
            and not key.startswith("tail_entity_biases.")
            and not key.startswith("relation_head_entity_biases.")
            and not key.startswith("relation_tail_entity_biases.")
        ]
        if relevant_missing or unexpected_keys:
            print(
                "Loaded checkpoint with non-strict state-dict matching: "
                f"missing={relevant_missing}, unexpected={unexpected_keys}"
            )
        print(f"Loaded checkpoint from {checkpoint_path}")

    validation_history: list[dict[str, float]] = []

    best_monitor: dict[str, float | int | str | dict] = {"MRR": float("-inf")}

    def epoch_callback(epoch: int, model_: MUGKGC, row: dict[str, float]) -> bool:
        if args.validation_frequency <= 0 or epoch % args.validation_frequency:
            return False
        monitored_split = args.early_stop_split if args.early_stop_mrr is not None else "validation"
        max_triples = args.validation_max_triples if monitored_split == "validation" else args.eval_max_triples
        validation_results = evaluate_all(
            model=model_,
            multi_factory=multi_factory,
            split=monitored_split,
            batch_size=args.eval_batch_size,
            max_triples=max_triples,
            use_tqdm=args.eval_progress,
            use_reciprocal_relations=args.use_reciprocal_evaluation,
            type_constraints=args.eval_type_constraints,
        )
        _print_results(f"Filtered {monitored_split} metrics after epoch {epoch}", validation_results)
        validation_history.append({"epoch": float(epoch), **_flatten_metrics(monitored_split, validation_results)})
        target_dataset = args.early_stop_dataset or multi_factory.get_dataset_names()[0]
        if target_dataset not in validation_results:
            raise ValueError(f"monitored dataset {target_dataset!r} was not evaluated.")
        monitored_mrr = validation_results[target_dataset]["MRR"]
        if monitored_mrr > float(best_monitor["MRR"]):
            best_monitor.update(
                {
                    "MRR": float(monitored_mrr),
                    "epoch": int(epoch),
                    "dataset": target_dataset,
                    "split": monitored_split,
                    "metrics": validation_results[target_dataset],
                }
            )
            if args.best_checkpoint_path:
                _save_checkpoint(
                    args.best_checkpoint_path,
                    model=model_,
                    args=args,
                    dataset_names=multi_factory.get_dataset_names(),
                    extra={
                        "best_epoch": int(epoch),
                        "best_split": monitored_split,
                        "best_dataset": target_dataset,
                        "best_metrics": validation_results[target_dataset],
                    },
                )
                print(
                    f"Saved best checkpoint to {args.best_checkpoint_path} "
                    f"({target_dataset} {monitored_split} MRR={monitored_mrr:.4f}, epoch={epoch})"
                )
        if args.early_stop_mrr is None:
            return False
        if monitored_mrr >= args.early_stop_mrr:
            print(
                f"Early stop target reached: {target_dataset} {monitored_split} "
                f"MRR={monitored_mrr:.4f} >= {args.early_stop_mrr:.4f}"
            )
            return True
        return False

    history = train_joint(
        model=model,
        multi_factory=multi_factory,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_negs_per_pos=args.num_negs_per_pos,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sampling_strategy=args.sampling_strategy,
        dataset_sampling_temperature=args.dataset_sampling_temperature,
        dataset_sampling_weights=_parse_dataset_weights(args.dataset_weights),
        steps_per_epoch=args.steps_per_epoch,
        device=args.device,
        loss_name=args.loss,
        adversarial_temperature=args.adversarial_temperature,
        margin=args.margin,
        training_mode=args.training_mode,
        lcwa_label_smoothing=args.lcwa_label_smoothing,
        lcwa_loss_name=args.lcwa_loss,
        aux_loss_weight=0.0 if args.disable_mug else args.aux_loss_weight,
        epoch_callback=epoch_callback,
    )

    if args.checkpoint_path:
        _save_checkpoint(
            args.checkpoint_path,
            model=model,
            args=args,
            dataset_names=multi_factory.get_dataset_names(),
        )
        print(f"\nSaved checkpoint to {args.checkpoint_path}")

    test_results = None
    if args.skip_evaluation:
        print("\nSkipping final test evaluation.")
    else:
        test_results = evaluate_all(
            model=model,
            multi_factory=multi_factory,
            split="testing",
            batch_size=args.eval_batch_size,
            max_triples=args.eval_max_triples,
            use_tqdm=args.eval_progress,
            use_reciprocal_relations=args.use_reciprocal_evaluation,
            type_constraints=args.eval_type_constraints,
        )
        _print_results("Filtered test metrics", test_results)

    if args.history_csv_path:
        rows = list(history) + validation_history
        _save_history_csv(args.history_csv_path, rows)
        print(f"\nSaved history CSV to {args.history_csv_path}")

    if args.result_json_path:
        _save_json(
            args.result_json_path,
            {
                "args": vars(args),
                "dataset_names": multi_factory.get_dataset_names(),
                "history": history,
                "validation_history": validation_history,
                "best_monitor": best_monitor if best_monitor["MRR"] != float("-inf") else None,
                "test_results": test_results,
            },
        )
        print(f"Saved result JSON to {args.result_json_path}")


if __name__ == "__main__":
    main()
