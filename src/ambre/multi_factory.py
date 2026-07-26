"""Wrappers for keeping multiple PyKEEN triples factories side-by-side."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from pykeen.triples import CoreTriplesFactory
from pykeen.typing import MappedTriples


@dataclass
class DatasetFactories:
    """The train/validation/test factories for one KG dataset."""

    training: CoreTriplesFactory
    validation: CoreTriplesFactory | None = None
    testing: CoreTriplesFactory | None = None


class MultiTriplesFactory:
    """A dataset-boundary preserving wrapper around multiple triples factories."""

    def __init__(self, factories: Mapping[str, DatasetFactories]):
        if not factories:
            raise ValueError("At least one dataset is required.")
        self.factories = dict(factories)

    def get_dataset_names(self) -> list[str]:
        """Return dataset names in deterministic insertion order."""
        return list(self.factories)

    def get_training_factory(self, dataset_name: str) -> CoreTriplesFactory:
        """Return the training triples factory for a dataset."""
        return self.factories[dataset_name].training

    def get_validation_factory(self, dataset_name: str) -> CoreTriplesFactory | None:
        """Return the validation triples factory for a dataset, if present."""
        return self.factories[dataset_name].validation

    def get_testing_factory(self, dataset_name: str) -> CoreTriplesFactory | None:
        """Return the testing triples factory for a dataset, if present."""
        return self.factories[dataset_name].testing

    def get_num_entities(self, dataset_name: str) -> int:
        """Return the number of entities for a dataset."""
        return self.get_training_factory(dataset_name).num_entities

    def get_num_relations(self, dataset_name: str) -> int:
        """Return the number of relations for a dataset."""
        return self.get_training_factory(dataset_name).num_relations

    def get_all_known_triples(self, dataset_name: str) -> MappedTriples:
        """Return train+valid+test triples for filtered evaluation in one dataset only."""
        dfs = self.factories[dataset_name]
        parts = [dfs.training.mapped_triples]
        if dfs.validation is not None:
            parts.append(dfs.validation.mapped_triples)
        if dfs.testing is not None:
            parts.append(dfs.testing.mapped_triples)
        return torch.cat(parts, dim=0)

    def iter_train_factories(self) -> Iterable[tuple[str, CoreTriplesFactory]]:
        """Iterate over ``(dataset_name, training_factory)`` pairs."""
        for name, dfs in self.factories.items():
            yield name, dfs.training

    def to(self, device: torch.device | str) -> "MultiTriplesFactory":
        """Move mapped triples to a device in-place and return ``self``."""
        for dfs in self.factories.values():
            dfs.training.mapped_triples = dfs.training.mapped_triples.to(device)
            if dfs.validation is not None:
                dfs.validation.mapped_triples = dfs.validation.mapped_triples.to(device)
            if dfs.testing is not None:
                dfs.testing.mapped_triples = dfs.testing.mapped_triples.to(device)
        return self


def multi_factory_from_datasets(datasets: Mapping[str, object]) -> MultiTriplesFactory:
    """Create a :class:`MultiTriplesFactory` from PyKEEN dataset instances."""
    return MultiTriplesFactory(
        factories={
            name: DatasetFactories(
                training=dataset.training,
                validation=getattr(dataset, "validation", None),
                testing=getattr(dataset, "testing", None),
            )
            for name, dataset in datasets.items()
        }
    )
