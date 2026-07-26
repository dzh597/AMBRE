"""Structural feature construction for AMBRE."""

from __future__ import annotations

import torch

from pykeen.typing import MappedTriples


def build_entity_features(
    mapped_triples: MappedTriples,
    num_entities: int,
    num_relations: int,
    mode: str = "relation_incidence",
    external_features: torch.FloatTensor | None = None,
) -> torch.FloatTensor:
    """Build entity features for a single KG.

    The MVP default is relation incidence:
    ``[head_count_per_relation, tail_count_per_relation, in_degree, out_degree, total_degree]``.
    """
    if mode == "external":
        if external_features is None:
            raise ValueError("external_features must be given for mode='external'.")
        if external_features.shape[0] != num_entities:
            raise ValueError("external_features must have one row per entity.")
        return external_features.float()

    if mode == "identity":
        return torch.eye(num_entities, dtype=torch.float, device=mapped_triples.device)

    if mode != "relation_incidence":
        raise ValueError(f"Unsupported entity feature mode: {mode}")

    device = mapped_triples.device
    features = torch.zeros(num_entities, 2 * num_relations + 3, dtype=torch.float, device=device)
    if mapped_triples.numel() == 0:
        return features

    heads = mapped_triples[:, 0]
    relations = mapped_triples[:, 1]
    tails = mapped_triples[:, 2]
    ones = torch.ones_like(relations, dtype=torch.float)

    # Relation-specific head/tail incidence counts.
    features.index_put_((heads, relations), ones, accumulate=True)
    features.index_put_((tails, num_relations + relations), ones, accumulate=True)

    # Degree features: in, out, total.
    in_degree_col = 2 * num_relations
    out_degree_col = in_degree_col + 1
    total_degree_col = out_degree_col + 1
    features.index_put_((tails, torch.full_like(tails, in_degree_col)), ones, accumulate=True)
    features.index_put_((heads, torch.full_like(heads, out_degree_col)), ones, accumulate=True)
    features[:, total_degree_col] = features[:, in_degree_col] + features[:, out_degree_col]

    # Simple per-column normalization stabilizes cross-KG optimization.
    scale = features.max(dim=0).values.clamp_min(1.0)
    return features / scale
