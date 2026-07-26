"""Minimal smoke tests for the anonymous AMBRE package."""

from __future__ import annotations

import numpy as np
import torch

from pykeen.triples import TriplesFactory

from ambre.encoder import SharedNonBacktrackingSpectralEncoder
from ambre.model import MUGKGC
from ambre.multi_factory import DatasetFactories, MultiTriplesFactory


def _tiny_factory() -> TriplesFactory:
    triples = np.asarray(
        [
            ["a", "r0", "b"],
            ["b", "r0", "c"],
            ["c", "r1", "a"],
            ["a", "r1", "c"],
        ],
        dtype=str,
    )
    return TriplesFactory.from_labeled_triples(triples=triples)


def _multi_factory() -> MultiTriplesFactory:
    factory = _tiny_factory()
    return MultiTriplesFactory({"tiny": DatasetFactories(training=factory)})


def test_non_backtracking_step_excludes_immediate_reversal() -> None:
    encoder = SharedNonBacktrackingSpectralEncoder(input_dim=2, hidden_dim=2, num_layers=1, dropout=0.0)
    src = torch.as_tensor([0, 1, 1], dtype=torch.long)
    dst = torch.as_tensor([1, 0, 2], dtype=torch.long)
    incoming, outgoing, lookup = encoder._build_node_edge_lists(src=src, dst=dst, num_nodes=3)

    previous = torch.as_tensor([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    propagated = encoder._non_backtracking_step(
        previous=previous,
        src=src,
        dst=dst,
        incoming=incoming,
        outgoing=outgoing,
        lookup=lookup,
    )

    assert torch.allclose(propagated[1], torch.zeros(2))
    assert torch.allclose(propagated[2], previous[0])


def test_model_scores_tiny_graph() -> None:
    torch.manual_seed(0)
    multi_factory = _multi_factory()
    model = MUGKGC(
        multi_factory=multi_factory,
        embedding_dim=8,
        nb_max_length=1,
        nb_top_k=2,
        nb_min_count=1,
        use_nb_encoder=True,
        cache_refresh_interval=2,
        view_refresh_size=1,
    )
    triples = multi_factory.get_training_factory("tiny").mapped_triples[:2]
    scores = model.score_hrt("tiny", triples)
    aux = model.get_auxiliary_loss("tiny")

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert aux.shape == torch.Size([])
    assert torch.isfinite(aux)

    loss = scores.sum() + aux
    loss.backward()


def test_cache_reuse_before_refresh_interval() -> None:
    multi_factory = _multi_factory()
    model = MUGKGC(
        multi_factory=multi_factory,
        embedding_dim=8,
        nb_max_length=1,
        nb_top_k=2,
        nb_min_count=1,
        use_nb_encoder=False,
        cache_refresh_interval=3,
    )

    calls = 0
    original_compute_all = model.entity_representations._compute_all

    def wrapped_compute_all(dataset_name: str):
        nonlocal calls
        calls += 1
        return original_compute_all(dataset_name)

    model.entity_representations._compute_all = wrapped_compute_all  # type: ignore[assignment]
    triples = multi_factory.get_training_factory("tiny").mapped_triples[:2]

    model.score_hrt("tiny", triples).sum().backward()
    assert calls == 1
    model.zero_grad(set_to_none=True)

    model.advance_cache_version()
    model.score_hrt("tiny", triples).sum().backward()
    assert calls == 1
