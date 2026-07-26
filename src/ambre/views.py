"""Sparse relation-sequence view construction for the non-backtracking encoder."""

from __future__ import annotations

from collections import Counter, defaultdict

import torch

from pykeen.typing import MappedTriples


def _sparse_adjacency(edges: list[tuple[int, int]], num_entities: int, device: torch.device) -> torch.Tensor:
    if not edges:
        indices = torch.empty(2, 0, dtype=torch.long, device=device)
        values = torch.empty(0, dtype=torch.float, device=device)
    else:
        indices = torch.as_tensor(edges, dtype=torch.long, device=device).t().contiguous()
        values = torch.ones(indices.shape[1], dtype=torch.float, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_entities, num_entities), device=device).coalesce()


def build_relation_sequence_views(
    mapped_triples: MappedTriples,
    num_entities: int,
    num_relations: int,
    max_path_length: int = 2,
    top_k_paths: int = 32,
    min_path_count: int = 10,
    max_two_hop_paths: int = 200_000,
    max_two_hop_paths_per_middle: int = 512,
    max_edges_per_view: int = 50_000,
) -> dict[str, torch.Tensor]:
    """Build sparse adjacency matrices for frequent relation-sequence views.

    The non-backtracking encoder consumes directed relation-sequence views. This
    builder supports length-1 and length-2 relation sequences. Length-2 paths
    are enumerated through a shared middle entity and kept by frequency.

    For larger KGs (e.g., FB15k-237), the number of two-hop paths can be very
    large. The ``max_*`` arguments cap the sampled sparse edges while preserving
    dataset boundaries and avoiding dense ``O(num_entities^2)`` adjacency.
    """
    if max_path_length < 1:
        raise ValueError("max_path_length must be positive.")
    device = mapped_triples.device
    triples = mapped_triples.detach().cpu().tolist()

    path_edges: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)

    # Length-1 relation views.
    relation_counts = Counter()
    for h, r, t in triples:
        key = (int(r),)
        relation_counts[key] += 1
        path_edges[key].append((int(h), int(t)))

    # Length-2 relation-sequence views. We sample/cap two-hop paths to avoid
    # materializing high-degree Cartesian products on larger KGs.
    if max_path_length >= 2:
        incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
        outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for h, r, t in triples:
            outgoing[int(h)].append((int(r), int(t)))
            incoming[int(t)].append((int(h), int(r)))

        sampled_two_hop = 0
        for middle in range(num_entities):
            in_edges = incoming.get(middle, [])
            out_edges = outgoing.get(middle, [])
            if not in_edges or not out_edges:
                continue

            pair_count = len(in_edges) * len(out_edges)
            if pair_count <= max_two_hop_paths_per_middle:
                for h, r1 in in_edges:
                    for r2, t in out_edges:
                        key = (int(r1), int(r2))
                        if len(path_edges[key]) < max_edges_per_view:
                            path_edges[key].append((int(h), int(t)))
                        sampled_two_hop += 1
                        if sampled_two_hop >= max_two_hop_paths:
                            break
                    if sampled_two_hop >= max_two_hop_paths:
                        break
            else:
                for _ in range(max_two_hop_paths_per_middle):
                    h, r1 = in_edges[torch.randint(high=len(in_edges), size=()).item()]
                    r2, t = out_edges[torch.randint(high=len(out_edges), size=()).item()]
                    key = (int(r1), int(r2))
                    if len(path_edges[key]) < max_edges_per_view:
                        path_edges[key].append((int(h), int(t)))
                    sampled_two_hop += 1

            if sampled_two_hop >= max_two_hop_paths:
                break

    # Select frequent relation sequences. If min_path_count is too strict on
    # tiny KGs, fall back to the most frequent sequences so the encoder always
    # has at least one view.
    counts = Counter({key: len(edges) for key, edges in path_edges.items()})
    selected = [(key, count) for key, count in counts.most_common() if count >= min_path_count]
    if not selected:
        selected = counts.most_common(top_k_paths)
    else:
        selected = selected[:top_k_paths]

    views: dict[str, torch.Tensor] = {}
    for key, _count in selected:
        if len(key) == 1:
            name = f"r{key[0]}"
        else:
            name = "->".join(f"r{r}" for r in key)
        views[name] = _sparse_adjacency(path_edges[key], num_entities=num_entities, device=device)

    if not views:
        # Degenerate empty graph fallback.
        views["empty"] = _sparse_adjacency([], num_entities=num_entities, device=device)
    return views

