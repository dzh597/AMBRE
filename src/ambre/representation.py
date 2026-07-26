"""Entity representation module for AMBRE."""

from __future__ import annotations

import re

import torch
from torch import nn

from .encoder import DimensionAwareAlignment, SharedNonBacktrackingSpectralEncoder, StandardFeatureProjection
from .features import build_entity_features
from .views import build_relation_sequence_views
from .multi_factory import MultiTriplesFactory


def _safe_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


class MUGEntityRepresentation(nn.Module):
    """Construct and serve per-dataset AMBRE entity representations."""

    def __init__(
        self,
        multi_factory: MultiTriplesFactory,
        embedding_dim: int,
        feature_mode: str = "relation_incidence",
        nb_max_length: int = 2,
        nb_top_k: int = 32,
        nb_min_count: int = 10,
        nb_max_two_hop_paths: int = 200_000,
        nb_max_two_hop_paths_per_middle: int = 512,
        nb_max_edges_per_view: int = 50_000,
        lambda_align: float = 0.01,
        lambda_recon: float = 0.1,
        lambda_scatter: float = 0.001,
        dropout: float = 0.1,
        use_dimension_alignment: bool = True,
        use_nb_encoder: bool = True,
        cache_refresh_interval: int = 4,
        view_refresh_size: int = 8,
        view_refresh_strategy: str = "rotate",
        structural_dataset_names: set[str] | None = None,
    ):
        super().__init__()
        if cache_refresh_interval < 1:
            raise ValueError("cache_refresh_interval must be positive.")
        if view_refresh_size < 1:
            raise ValueError("view_refresh_size must be positive.")
        if view_refresh_strategy not in {"all", "rotate", "sample"}:
            raise ValueError("view_refresh_strategy must be 'all', 'rotate', or 'sample'.")
        self.multi_factory = multi_factory
        self.embedding_dim = embedding_dim
        self.lambda_align = lambda_align
        self.lambda_recon = lambda_recon
        self.lambda_scatter = lambda_scatter
        self.use_nb_encoder = use_nb_encoder
        self.cache_refresh_interval = cache_refresh_interval
        self.view_refresh_size = view_refresh_size
        self.view_refresh_strategy = view_refresh_strategy
        self.structural_dataset_names = set(structural_dataset_names or multi_factory.get_dataset_names())
        self.name_to_key = {name: _safe_key(name) for name in multi_factory.get_dataset_names()}
        self.key_to_name = {key: name for name, key in self.name_to_key.items()}

        self.feature_projections = nn.ModuleDict()
        self.features: dict[str, torch.FloatTensor] = {}
        self.nb_view_adjs: dict[str, dict[str, torch.Tensor]] = {}
        for dataset_name, factory in multi_factory.iter_train_factories():
            if dataset_name not in self.structural_dataset_names:
                continue
            key = self.name_to_key[dataset_name]
            features = build_entity_features(
                mapped_triples=factory.mapped_triples,
                num_entities=factory.num_entities,
                num_relations=factory.num_relations,
                mode=feature_mode,
            )
            self.features[dataset_name] = features
            projection_cls = DimensionAwareAlignment if use_dimension_alignment else StandardFeatureProjection
            self.feature_projections[key] = projection_cls(input_dim=features.shape[1], output_dim=embedding_dim)
            self.nb_view_adjs[dataset_name] = build_relation_sequence_views(
                mapped_triples=factory.mapped_triples,
                num_entities=factory.num_entities,
                num_relations=factory.num_relations,
                max_path_length=nb_max_length,
                top_k_paths=nb_top_k,
                min_path_count=nb_min_count,
                max_two_hop_paths=nb_max_two_hop_paths,
                max_two_hop_paths_per_middle=nb_max_two_hop_paths_per_middle,
                max_edges_per_view=nb_max_edges_per_view,
            )

        self.encoder = SharedNonBacktrackingSpectralEncoder(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_layers=2,
            dropout=dropout,
        )
        self._model_version = 0
        self._cache: dict[str, torch.FloatTensor] = {}
        self._aux_cache: dict[str, torch.FloatTensor] = {}
        self._pending_cache: dict[str, torch.FloatTensor] = {}
        self._pending_aux_cache: dict[str, torch.FloatTensor] = {}
        self._cache_version: dict[str, int] = {}
        self._pending_version: dict[str, int] = {}
        self._cache_stale: dict[str, bool] = {}
        self._cache_stale_since_version: dict[str, int] = {}
        self._view_names: dict[str, list[str]] = {
            dataset_name: list(views)
            for dataset_name, views in self.nb_view_adjs.items()
        }
        self._view_cursor: dict[str, int] = {dataset_name: 0 for dataset_name in self.nb_view_adjs}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def clear_cache(self) -> None:
        """Clear cached entity representations and auxiliary losses."""
        self._cache.clear()
        self._aux_cache.clear()
        self._pending_cache.clear()
        self._pending_aux_cache.clear()
        self._cache_version.clear()
        self._pending_version.clear()
        self._cache_stale.clear()
        self._cache_stale_since_version.clear()

    def advance_cache_version(self) -> None:
        """Advance the global model version and mark cached outputs stale."""
        self._model_version += 1
        self._pending_cache.clear()
        self._pending_aux_cache.clear()
        self._pending_version.clear()
        for dataset_name in self.structural_dataset_names:
            if not self._cache_stale.get(dataset_name, False):
                self._cache_stale[dataset_name] = True
                self._cache_stale_since_version[dataset_name] = self._cache_version.get(
                    dataset_name,
                    self._model_version,
                )

    def _needs_refresh(self, dataset_name: str) -> bool:
        if dataset_name not in self.features:
            raise KeyError(f"Structural entity features are disabled for dataset: {dataset_name}")
        if dataset_name not in self._cache:
            return True
        if not self._cache_stale.get(dataset_name, False):
            return False
        stale_since_version = self._cache_stale_since_version.get(
            dataset_name,
            self._cache_version.get(dataset_name, self._model_version),
        )
        return self._model_version - stale_since_version >= self.cache_refresh_interval

    def _store_cache(
        self,
        dataset_name: str,
        z: torch.FloatTensor,
        aux: torch.FloatTensor,
    ) -> None:
        self._cache[dataset_name] = z.detach()
        self._aux_cache[dataset_name] = aux.detach()
        self._cache_version[dataset_name] = self._model_version
        self._pending_cache[dataset_name] = z
        self._pending_aux_cache[dataset_name] = aux
        self._pending_version[dataset_name] = self._model_version
        self._cache_stale[dataset_name] = False
        self._cache_stale_since_version.pop(dataset_name, None)

    def _select_view_names(self, dataset_name: str) -> list[str]:
        view_names = self._view_names[dataset_name]
        if not view_names:
            return []
        if not self.training or self.view_refresh_strategy == "all":
            return view_names
        if len(view_names) <= self.view_refresh_size:
            return view_names
        if self.view_refresh_strategy == "sample":
            perm = torch.randperm(len(view_names), device="cpu")[: self.view_refresh_size].tolist()
            return [view_names[index] for index in perm]
        start = self._view_cursor.get(dataset_name, 0) % len(view_names)
        count = min(self.view_refresh_size, len(view_names))
        selected = [view_names[(start + offset) % len(view_names)] for offset in range(count)]
        self._view_cursor[dataset_name] = (start + count) % len(view_names)
        return selected

    def _compute_all(self, dataset_name: str) -> torch.FloatTensor:
        key = self.name_to_key[dataset_name]
        x = self.features[dataset_name].to(device=self.device)
        aligned, align_loss = self.feature_projections[key](x)
        if self.use_nb_encoder:
            view_names = self._select_view_names(dataset_name)
            view_adjs = {view_name: self.nb_view_adjs[dataset_name][view_name] for view_name in view_names}
            z, recon_loss, scatter_loss = self.encoder(aligned, view_adjs)
        else:
            z = aligned
            recon_loss = z.new_zeros(())
            center = z.mean(dim=0, keepdim=True)
            scatter = (z - center).pow(2).sum(dim=-1).mean()
            scatter_loss = -scatter / (1.0 + scatter.detach())
        aux = (
            self.lambda_align * align_loss
            + self.lambda_recon * recon_loss
            + self.lambda_scatter * scatter_loss
        )
        self._store_cache(dataset_name=dataset_name, z=z, aux=aux)
        return z

    def get_all_entity_representations(self, dataset_name: str) -> torch.FloatTensor:
        """Return all entity representations for one dataset."""
        if self._needs_refresh(dataset_name):
            return self._compute_all(dataset_name)
        if self._pending_version.get(dataset_name) == self._model_version:
            return self._pending_cache[dataset_name]
        return self._cache[dataset_name]

    def forward(self, dataset_name: str, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        """Return representations for selected entity IDs in a dataset."""
        return self.get_all_entity_representations(dataset_name)[entity_ids.to(device=self.device)]

    def get_auxiliary_loss(self, dataset_name: str) -> torch.FloatTensor:
        """Return the cached/forced auxiliary loss for one dataset."""
        if self._needs_refresh(dataset_name):
            self._compute_all(dataset_name)
        if self._pending_version.get(dataset_name) == self._model_version:
            return self._pending_aux_cache[dataset_name]
        return self._aux_cache[dataset_name]

    def get_total_auxiliary_loss(self) -> torch.FloatTensor:
        """Return the sum of auxiliary losses across all datasets."""
        losses = [self.get_auxiliary_loss(name) for name in self.structural_dataset_names]
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).sum()
