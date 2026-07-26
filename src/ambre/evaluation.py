"""Separate filtered evaluation for each KG dataset."""

from __future__ import annotations

from collections import defaultdict

import torch
from tqdm.auto import tqdm

from .model import MUGKGC
from .multi_factory import MultiTriplesFactory


def _build_filter_maps(
    known_triples: torch.LongTensor,
    *,
    desc: str | None = None,
    use_tqdm: bool = False,
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    known_triples = known_triples.detach().cpu()
    if use_tqdm:
        chunk_size = 100_000
        with tqdm(
            total=known_triples.shape[0],
            desc=desc,
            unit="triple",
            unit_scale=True,
            leave=False,
        ) as progress:
            for start in range(0, known_triples.shape[0], chunk_size):
                chunk = known_triples[start : start + chunk_size]
                for h, r, t in chunk.tolist():
                    tails[(int(h), int(r))].add(int(t))
                    heads[(int(r), int(t))].add(int(h))
                progress.update(chunk.shape[0])
        return tails, heads

    for h, r, t in known_triples.tolist():
        tails[(int(h), int(r))].add(int(t))
        heads[(int(r), int(t))].add(int(h))
    return tails, heads


def _build_relation_type_masks(
    triples: torch.LongTensor,
    *,
    num_relations: int,
    num_entities: int,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """Return allowed heads/tails per relation from observed triples."""
    head_mask = torch.zeros(num_relations, num_entities, dtype=torch.bool)
    tail_mask = torch.zeros(num_relations, num_entities, dtype=torch.bool)
    for h, r, t in triples.detach().cpu().tolist():
        if int(r) < num_relations:
            head_mask[int(r), int(h)] = True
            tail_mask[int(r), int(t)] = True
    return head_mask, tail_mask


def _rank_from_scores(scores: torch.FloatTensor, true_index: int) -> int:
    scores = torch.nan_to_num(scores, nan=-torch.inf)
    true_score = scores[true_index]
    if not torch.isfinite(true_score):
        return scores.shape[0]
    return int(((scores > true_score).sum() + (scores == true_score).sum()).item())


def _get_eval_factory(multi_factory: MultiTriplesFactory, dataset_name: str, split: str):
    if split == "testing":
        return multi_factory.get_testing_factory(dataset_name)
    if split == "validation":
        return multi_factory.get_validation_factory(dataset_name)
    if split == "training":
        return multi_factory.get_training_factory(dataset_name)
    raise ValueError(f"Invalid evaluation split: {split!r}")


def evaluate_dataset(
    model: MUGKGC,
    multi_factory: MultiTriplesFactory,
    dataset_name: str,
    split: str = "testing",
    batch_size: int = 128,
    max_triples: int | None = None,
    use_tqdm: bool = False,
    use_reciprocal_relations: bool = False,
    type_constraints: str = "none",
) -> dict[str, float]:
    """Evaluate one dataset with candidates/filters from this dataset only."""
    eval_factory = _get_eval_factory(multi_factory=multi_factory, dataset_name=dataset_name, split=split)
    if eval_factory is None:
        raise ValueError(f"Dataset {dataset_name} has no {split} factory.")

    dfs = multi_factory.factories[dataset_name]
    known_parts = [dfs.training.mapped_triples]
    if split in {"validation", "testing"} and dfs.validation is not None:
        known_parts.append(dfs.validation.mapped_triples)
    if split == "testing" and dfs.testing is not None:
        known_parts.append(dfs.testing.mapped_triples)
    known = torch.cat(known_parts, dim=0)
    tail_filters, head_filters = _build_filter_maps(
        known,
        desc=f"{split}/{dataset_name} filters",
        use_tqdm=use_tqdm,
    )
    ranks: list[int] = []
    eval_triples = eval_factory.mapped_triples.detach().cpu()
    if max_triples is not None:
        eval_triples = eval_triples[:max_triples]

    head_type_mask = tail_type_mask = None
    if type_constraints != "none":
        if type_constraints == "training":
            type_parts = [dfs.training.mapped_triples]
        elif type_constraints == "training_validation":
            type_parts = [dfs.training.mapped_triples]
            if dfs.validation is not None:
                type_parts.append(dfs.validation.mapped_triples)
        else:
            raise ValueError("type_constraints must be 'none', 'training', or 'training_validation'.")
        num_relations_for_eval = max(int(eval_triples[:, 1].max().item()) + 1, 1) if eval_triples.numel() else 1
        head_type_mask, tail_type_mask = _build_relation_type_masks(
            torch.cat(type_parts, dim=0),
            num_relations=num_relations_for_eval,
            num_entities=multi_factory.get_num_entities(dataset_name),
        )

    model.eval()
    model.clear_cache()
    with torch.no_grad():
        starts = range(0, eval_triples.shape[0], batch_size)
        if use_tqdm:
            starts = tqdm(
                starts,
                desc=f"{split}/{dataset_name} ranking",
                total=(eval_triples.shape[0] + batch_size - 1) // batch_size,
                unit="batch",
                leave=True,
            )
        for start in starts:
            batch = eval_triples[start : start + batch_size]
            batch_list = [(int(h), int(r), int(t)) for h, r, t in batch.tolist()]

            hr_batch = torch.as_tensor(
                [[h, r] for h, r, _t in batch_list],
                dtype=torch.long,
                device=model.device,
            )
            tail_scores_batch = model.score_t(dataset_name, hr_batch).detach().clone()
            for row, (h, r, t) in enumerate(batch_list):
                if tail_type_mask is not None and r < tail_type_mask.shape[0]:
                    true_score = tail_scores_batch[row, t].clone()
                    allowed = tail_type_mask[r].to(device=tail_scores_batch.device)
                    tail_scores_batch[row, ~allowed] = -torch.inf
                    tail_scores_batch[row, t] = true_score
                for filtered_tail in tail_filters[(h, r)]:
                    if filtered_tail != t:
                        tail_scores_batch[row, filtered_tail] = -torch.inf
                ranks.append(_rank_from_scores(tail_scores_batch[row], t))

            if use_reciprocal_relations:
                relation_offset = multi_factory.get_num_relations(dataset_name) // 2
                hr_inverse_batch = torch.as_tensor(
                    [[t, r + relation_offset] for _h, r, t in batch_list],
                    dtype=torch.long,
                    device=model.device,
                )
                head_scores_batch = model.score_t(dataset_name, hr_inverse_batch).detach().clone()
            else:
                rt_batch = torch.as_tensor(
                    [[r, t] for _h, r, t in batch_list],
                    dtype=torch.long,
                    device=model.device,
                )
                head_scores_batch = model.score_h(dataset_name, rt_batch).detach().clone()
            for row, (h, r, t) in enumerate(batch_list):
                if head_type_mask is not None and r < head_type_mask.shape[0]:
                    true_score = head_scores_batch[row, h].clone()
                    allowed = head_type_mask[r].to(device=head_scores_batch.device)
                    head_scores_batch[row, ~allowed] = -torch.inf
                    # Keep the true entity eligible even if this exact entity
                    # was unseen for the relation in train/validation.
                    head_scores_batch[row, h] = true_score
                for filtered_head in head_filters[(r, t)]:
                    if filtered_head != h:
                        head_scores_batch[row, filtered_head] = -torch.inf
                ranks.append(_rank_from_scores(head_scores_batch[row], h))
    model.clear_cache()

    ranks_t = torch.as_tensor(ranks, dtype=torch.float)
    return {
        "MRR": float((1.0 / ranks_t).mean().item()),
        "Hits@1": float((ranks_t <= 1).float().mean().item()),
        "Hits@3": float((ranks_t <= 3).float().mean().item()),
        "Hits@10": float((ranks_t <= 10).float().mean().item()),
    }


def evaluate_all(
    model: MUGKGC,
    multi_factory: MultiTriplesFactory,
    *,
    split: str = "testing",
    batch_size: int = 128,
    max_triples: int | None = None,
    use_tqdm: bool = False,
    use_reciprocal_relations: bool = False,
    type_constraints: str = "none",
) -> dict[str, dict[str, float]]:
    """Evaluate all datasets separately and add a macro average."""
    results = {
        dataset_name: evaluate_dataset(
            model=model,
            multi_factory=multi_factory,
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            max_triples=max_triples,
            use_tqdm=use_tqdm,
            use_reciprocal_relations=use_reciprocal_relations,
            type_constraints=type_constraints,
        )
        for dataset_name in multi_factory.get_dataset_names()
    }
    metric_names = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    results["macro_average"] = {
        metric: sum(results[name][metric] for name in multi_factory.get_dataset_names())
        / len(multi_factory.get_dataset_names())
        for metric in metric_names
    }
    return results
