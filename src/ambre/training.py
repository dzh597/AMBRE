"""Custom joint multi-KG training loop for the AMBRE MVP."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import torch
from torch.nn import functional
from tqdm.auto import tqdm

from .model import MUGKGC
from .multi_factory import MultiTriplesFactory
from .sampler import BalancedDatasetBatchSampler


def kgc_negative_sampling_loss(
    positive_scores: torch.FloatTensor,
    negative_scores: torch.FloatTensor,
    *,
    loss_name: str = "softplus",
    adversarial_temperature: float = 1.0,
    margin: float = 9.0,
) -> torch.FloatTensor:
    """Compute an sLCWA loss for positive scores and per-positive negatives."""
    if loss_name == "softplus":
        return functional.softplus(-positive_scores).mean() + functional.softplus(negative_scores).mean()

    if loss_name == "adversarial-bce":
        positive_loss = functional.binary_cross_entropy_with_logits(
            positive_scores,
            torch.ones_like(positive_scores),
            reduction="mean",
        )
        negative_loss = functional.binary_cross_entropy_with_logits(
            negative_scores,
            torch.zeros_like(negative_scores),
            reduction="none",
        )
        negative_weights = negative_scores.detach().mul(adversarial_temperature).softmax(dim=-1)
        return positive_loss + (negative_weights * negative_loss).sum(dim=-1).mean()

    if loss_name == "nssa":
        positive_loss = -functional.logsigmoid(margin + positive_scores).mean()
        negative_loss = -functional.logsigmoid(-negative_scores - margin)
        negative_weights = negative_scores.detach().mul(adversarial_temperature).softmax(dim=-1)
        return positive_loss + (negative_weights * negative_loss).sum(dim=-1).mean()

    raise ValueError(f"Unsupported loss_name: {loss_name!r}")


def kgc_lcwa_loss(
    scores: torch.FloatTensor,
    tails: torch.LongTensor,
    *,
    label_smoothing: float = 0.0,
    loss_name: str = "bce",
    all_positive_tails: list[torch.LongTensor] | None = None,
) -> torch.FloatTensor:
    """Compute 1-vs-all tail prediction BCE loss for an LCWA-style batch."""
    if loss_name == "ce":
        return functional.cross_entropy(
            scores,
            tails.to(device=scores.device),
            label_smoothing=label_smoothing,
        )
    labels = torch.zeros_like(scores)
    if all_positive_tails is None:
        rows = torch.arange(scores.shape[0], device=scores.device)
        labels[rows, tails.to(device=scores.device)] = 1.0
    else:
        for row, positive_tails in enumerate(all_positive_tails):
            labels[row, positive_tails.to(device=scores.device)] = 1.0
    if loss_name == "mce":
        positive_scores = scores.masked_fill(labels <= 0.0, -torch.inf)
        loss = scores.logsumexp(dim=-1) - positive_scores.logsumexp(dim=-1)
        if label_smoothing:
            log_probs = scores.log_softmax(dim=-1)
            uniform_loss = -log_probs.mean(dim=-1)
            loss = (1.0 - label_smoothing) * loss + label_smoothing * uniform_loss
        return loss.mean()
    if label_smoothing:
        labels = labels * (1.0 - label_smoothing) + label_smoothing / scores.shape[1]
    if loss_name != "bce":
        raise ValueError(f"Unsupported LCWA loss_name: {loss_name!r}")
    return functional.binary_cross_entropy_with_logits(scores, labels)


def _build_lcwa_tail_index(multi_factory: MultiTriplesFactory) -> dict[str, dict[tuple[int, int], torch.LongTensor]]:
    """Build ``(head, relation) -> all true tails`` maps for LCWA multi-hot labels."""
    result: dict[str, dict[tuple[int, int], torch.LongTensor]] = {}
    for dataset_name in multi_factory.get_dataset_names():
        tails: dict[tuple[int, int], set[int]] = defaultdict(set)
        triples = multi_factory.get_training_factory(dataset_name).mapped_triples.detach().cpu()
        for h, r, t in triples.tolist():
            tails[(int(h), int(r))].add(int(t))
        result[dataset_name] = {
            key: torch.as_tensor(sorted(value), dtype=torch.long)
            for key, value in tails.items()
        }
    return result


def _build_lcwa_hr_pairs(
    tail_index: dict[str, dict[tuple[int, int], torch.LongTensor]],
) -> dict[str, torch.LongTensor]:
    """Build tensors of unique ``(head, relation)`` pairs for LCWA training."""
    return {
        dataset_name: torch.as_tensor(sorted(mapping), dtype=torch.long)
        for dataset_name, mapping in tail_index.items()
    }


def train_joint(
    model: MUGKGC,
    multi_factory: MultiTriplesFactory,
    *,
    num_epochs: int = 5,
    batch_size: int = 256,
    num_negs_per_pos: int = 1,
    lr: float = 1.0e-3,
    weight_decay: float = 0.0,
    sampling_strategy: str = "balanced",
    dataset_sampling_temperature: float = 1.0,
    dataset_sampling_weights: dict[str, float] | None = None,
    steps_per_epoch: int | None = None,
    device: torch.device | str | None = None,
    loss_name: str = "softplus",
    adversarial_temperature: float = 1.0,
    margin: float = 9.0,
    training_mode: str = "slcwa",
    lcwa_label_smoothing: float = 0.0,
    lcwa_loss_name: str = "bce",
    aux_loss_weight: float = 1.0,
    epoch_callback: Callable[[int, MUGKGC, dict[str, float]], bool | None] | None = None,
) -> list[dict[str, float]]:
    """Train one AMBRE model jointly over all datasets."""
    if training_mode not in {"slcwa", "lcwa", "lcwa-ce"}:
        raise ValueError("training_mode must be 'slcwa', 'lcwa', or 'lcwa-ce'.")
    if device is not None:
        model.to(device)
    if training_mode in {"lcwa", "lcwa-ce"}:
        # LCWA scores against all entities, so entity embedding gradients are
        # dense. However, relation/entity bias lookups still use sparse
        # embeddings, so keep them on SparseAdam and optimize the remaining
        # dense parameters with Adam.
        sparse_params = list(model.get_sparse_parameters(include_entity_embeddings=False))
        sparse_param_ids = {id(parameter) for parameter in sparse_params}
        dense_params = [parameter for parameter in model.parameters() if id(parameter) not in sparse_param_ids]
        optimizer = torch.optim.Adam(dense_params, lr=lr, weight_decay=weight_decay)
        sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=lr) if sparse_params else None
    else:
        sparse_params = list(model.get_sparse_parameters())
        sparse_param_ids = {id(parameter) for parameter in sparse_params}
        dense_params = [parameter for parameter in model.parameters() if id(parameter) not in sparse_param_ids]
        optimizer = torch.optim.Adam(dense_params, lr=lr, weight_decay=weight_decay)
        sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=lr)
    sampler = BalancedDatasetBatchSampler(
        multi_factory=multi_factory,
        batch_size=batch_size,
        num_negs_per_pos=num_negs_per_pos,
        strategy=sampling_strategy,
        dataset_sampling_temperature=dataset_sampling_temperature,
        dataset_sampling_weights=dataset_sampling_weights,
    )
    if steps_per_epoch is None:
        steps_per_epoch = len(sampler)
    lcwa_tail_index = _build_lcwa_tail_index(multi_factory) if training_mode == "lcwa" else None
    lcwa_hr_pairs = _build_lcwa_hr_pairs(lcwa_tail_index) if lcwa_tail_index is not None else None

    history: list[dict[str, float]] = []
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="epochs", unit="epoch", position=0, leave=True, dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)

        step_bar = tqdm(
            range(steps_per_epoch),
            desc=f"epoch {epoch:03d}",
            unit="step",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )
        for _step in step_bar:
            forced_dataset_name = None
            # Keep the default sampling balanced, but make short smoke tests
            # report every dataset at least once when possible.
            if sampling_strategy == "balanced" and steps_per_epoch >= len(sampler.dataset_names):
                forced_dataset_name = sampler.dataset_names[_step % len(sampler.dataset_names)]
            if training_mode == "lcwa" and lcwa_hr_pairs is not None and lcwa_tail_index is not None:
                batch = sampler.sample_lcwa_batch(hr_pairs=lcwa_hr_pairs, dataset_name=forced_dataset_name)
                dataset_name = str(batch["dataset_name"])
                hr_batch = batch["hr_batch"]
                positives = None
                negatives = None
            else:
                batch = sampler.sample_batch(dataset_name=forced_dataset_name)
                dataset_name = str(batch["dataset_name"])
                positives = batch["positives"]
                negatives = batch["negatives"]

            optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad(set_to_none=True)
            if training_mode == "lcwa":
                scores = model.score_t(dataset_name, hr_batch)
                all_positive_tails = None
                if lcwa_tail_index is not None:
                    dataset_tail_index = lcwa_tail_index[dataset_name]
                    all_positive_tails = [
                        dataset_tail_index[(int(h), int(r))]
                        for h, r in hr_batch.detach().cpu().tolist()
                    ]
                tails = torch.as_tensor(
                    [int(positive_tails[0]) for positive_tails in all_positive_tails],
                    dtype=torch.long,
                    device=model.device,
                )
                kgc_loss = kgc_lcwa_loss(
                    scores=scores,
                    tails=tails,
                    label_smoothing=lcwa_label_smoothing,
                    loss_name=lcwa_loss_name,
                    all_positive_tails=all_positive_tails,
                )
            elif training_mode == "lcwa-ce":
                scores = model.score_t(dataset_name, positives[:, :2])
                kgc_loss = kgc_lcwa_loss(
                    scores=scores,
                    tails=positives[:, 2],
                    label_smoothing=lcwa_label_smoothing,
                    loss_name="ce",
                )
            else:
                positive_scores = model.score_hrt(dataset_name, positives)
                negative_scores = model.score_hrt(dataset_name, negatives.view(-1, 3)).view(negatives.shape[:-1])
                kgc_loss = kgc_negative_sampling_loss(
                    positive_scores,
                    negative_scores,
                    loss_name=loss_name,
                    adversarial_temperature=adversarial_temperature,
                    margin=margin,
                )
            aux_loss = model.get_auxiliary_loss(dataset_name)
            loss = kgc_loss + aux_loss_weight * aux_loss
            loss.backward()
            optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            model.advance_cache_version()

            loss_value = float(loss.detach().cpu())
            kgc_loss_value = float(kgc_loss.detach().cpu())
            aux_loss_value = float(aux_loss.detach().cpu())
            postfix_interval = max(1, steps_per_epoch // 200)
            if _step == 0 or (_step + 1) == steps_per_epoch or (_step + 1) % postfix_interval == 0:
                step_bar.set_postfix(
                    dataset=dataset_name,
                    kgc=f"{kgc_loss_value:.4f}",
                    aux=f"{aux_loss_value:.4f}",
                    loss=f"{loss_value:.4f}",
                )

            totals[f"{dataset_name}.loss"] += loss_value
            totals[f"{dataset_name}.kgc_loss"] += kgc_loss_value
            totals[f"{dataset_name}.aux_loss"] += aux_loss_value
            counts[dataset_name] += 1

        row: dict[str, float] = {"epoch": float(epoch)}
        for dataset_name in multi_factory.get_dataset_names():
            for suffix in ("loss", "kgc_loss", "aux_loss"):
                key = f"{dataset_name}.{suffix}"
                row[key] = totals[key] / counts[dataset_name] if counts[dataset_name] else float("nan")
        history.append(row)
        metrics = " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch")
        epoch_bar.set_postfix_str(metrics)
        tqdm.write(f"[epoch {epoch:03d}] {metrics}")
        if epoch_callback is not None:
            should_stop = bool(epoch_callback(epoch, model, row))
            if should_stop:
                tqdm.write(f"Stopping early after epoch {epoch:03d}.")
                break
    epoch_bar.close()
    return history
