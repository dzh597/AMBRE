"""Dataset-aware positive batching and negative sampling."""

from __future__ import annotations

import math
import random
from collections import deque

import torch

from .multi_factory import MultiTriplesFactory


class DatasetAwareNegativeSampler:
    """Corrupt heads/tails only inside the current dataset's entity space."""

    def __init__(self, num_negs_per_pos: int = 1):
        self.num_negs_per_pos = num_negs_per_pos

    def sample(self, positive_batch: torch.LongTensor, num_entities: int) -> torch.LongTensor:
        """Return negative triples shaped ``(batch_size, num_negs_per_pos, 3)``."""
        batch_size = positive_batch.shape[0]
        negative_batch = positive_batch.unsqueeze(1).repeat(1, self.num_negs_per_pos, 1)
        flat = negative_batch.view(-1, 3)
        total = flat.shape[0]
        corrupt_head = torch.rand(total, device=positive_batch.device) < 0.5
        replacement = torch.randint(
            high=max(num_entities - 1, 1),
            size=(total,),
            device=positive_batch.device,
        )
        if num_entities > 1:
            head_original = flat[:, 0]
            tail_original = flat[:, 2]
            replacement_head = replacement + (replacement >= head_original).long()
            replacement_tail = replacement + (replacement >= tail_original).long()
            flat[corrupt_head, 0] = replacement_head[corrupt_head]
            flat[~corrupt_head, 2] = replacement_tail[~corrupt_head]
        return flat.view(batch_size, self.num_negs_per_pos, 3)


class BalancedDatasetBatchSampler:
    """Uniformly sample a dataset first, then a positive batch from that dataset."""

    def __init__(
        self,
        multi_factory: MultiTriplesFactory,
        batch_size: int,
        num_negs_per_pos: int = 1,
        strategy: str = "balanced",
        dataset_sampling_temperature: float = 1.0,
        dataset_sampling_weights: dict[str, float] | None = None,
    ):
        if strategy not in {"balanced", "proportional", "temperature", "weighted"}:
            raise ValueError("strategy must be 'balanced', 'proportional', 'temperature', or 'weighted'.")
        self.multi_factory = multi_factory
        self.batch_size = batch_size
        self.strategy = strategy
        self.dataset_sampling_temperature = dataset_sampling_temperature
        self.negative_sampler = DatasetAwareNegativeSampler(num_negs_per_pos=num_negs_per_pos)
        self.dataset_names = multi_factory.get_dataset_names()
        self.dataset_sampling_weights = dataset_sampling_weights
        if self.dataset_sampling_weights is not None:
            unknown = sorted(set(self.dataset_sampling_weights).difference(self.dataset_names))
            if unknown:
                raise ValueError(f"Unknown dataset_sampling_weights entries: {unknown}")
            missing = sorted(set(self.dataset_names).difference(self.dataset_sampling_weights))
            if missing:
                raise ValueError(f"Missing dataset_sampling_weights entries: {missing}")
            if any(weight < 0.0 for weight in self.dataset_sampling_weights.values()):
                raise ValueError("dataset_sampling_weights must be non-negative.")
            if sum(self.dataset_sampling_weights.values()) <= 0.0:
                raise ValueError("At least one dataset_sampling_weights value must be positive.")
        self._triple_ids: dict[str, deque[int]] = {}
        self._hr_pair_ids: dict[str, deque[int]] = {}

    def __len__(self) -> int:
        """Use one proportional pass over all training triples as epoch length."""
        return sum(
            math.ceil(self.multi_factory.get_training_factory(name).mapped_triples.shape[0] / self.batch_size)
            for name in self.dataset_names
        )

    def _choose_dataset(self) -> str:
        if self.dataset_sampling_weights is not None or self.strategy == "weighted":
            if self.dataset_sampling_weights is None:
                raise ValueError("strategy='weighted' requires dataset_sampling_weights.")
            weights = [self.dataset_sampling_weights[name] for name in self.dataset_names]
            return random.choices(self.dataset_names, weights=weights, k=1)[0]
        if self.strategy == "balanced":
            return random.choice(self.dataset_names)
        if self.strategy == "proportional":
            weights = [
                self.multi_factory.get_training_factory(name).mapped_triples.shape[0]
                for name in self.dataset_names
            ]
        else:
            weights = [
                self.multi_factory.get_training_factory(name).mapped_triples.shape[0] ** self.dataset_sampling_temperature
                for name in self.dataset_names
            ]
        return random.choices(self.dataset_names, weights=weights, k=1)[0]

    def _next_ids(self, dataset_name: str) -> list[int]:
        factory = self.multi_factory.get_training_factory(dataset_name)
        num_triples = factory.mapped_triples.shape[0]
        queue = self._triple_ids.get(dataset_name)
        if queue is None or len(queue) < self.batch_size:
            ids = torch.randperm(num_triples).tolist()
            if queue is None:
                queue = deque()
            queue.extend(ids)
            self._triple_ids[dataset_name] = queue
        return [queue.popleft() for _ in range(min(self.batch_size, len(queue)))]

    def _next_hr_pair_ids(self, dataset_name: str, *, num_pairs: int) -> list[int]:
        queue = self._hr_pair_ids.get(dataset_name)
        if queue is None or len(queue) < self.batch_size:
            ids = torch.randperm(num_pairs).tolist()
            if queue is None:
                queue = deque()
            queue.extend(ids)
            self._hr_pair_ids[dataset_name] = queue
        return [queue.popleft() for _ in range(min(self.batch_size, len(queue)))]

    def sample_batch(self, dataset_name: str | None = None) -> dict[str, object]:
        """Sample one dataset-aware sLCWA-style batch."""
        dataset_name = dataset_name or self._choose_dataset()
        factory = self.multi_factory.get_training_factory(dataset_name)
        ids = self._next_ids(dataset_name)
        positives = factory.mapped_triples[ids]
        negatives = self.negative_sampler.sample(
            positive_batch=positives,
            num_entities=factory.num_entities,
        )
        return {
            "dataset_name": dataset_name,
            "positives": positives,
            "negatives": negatives,
        }

    def sample_lcwa_batch(
        self,
        hr_pairs: dict[str, torch.LongTensor],
        dataset_name: str | None = None,
    ) -> dict[str, object]:
        """Sample a dataset-aware LCWA batch of unique ``(head, relation)`` pairs."""
        dataset_name = dataset_name or self._choose_dataset()
        pairs = hr_pairs[dataset_name]
        ids = self._next_hr_pair_ids(dataset_name, num_pairs=pairs.shape[0])
        return {
            "dataset_name": dataset_name,
            "hr_batch": pairs[ids],
        }
