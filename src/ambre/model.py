"""AMBRE model with configurable shallow KGE scoring."""

from __future__ import annotations

import math

import torch
from torch import nn

from .multi_factory import MultiTriplesFactory
from .representation import MUGEntityRepresentation, _safe_key


class MUGKGC(nn.Module):
    """A minimal multi-KG AMBRE model for joint training and separate ranking."""

    def __init__(
        self,
        multi_factory: MultiTriplesFactory,
        embedding_dim: int = 128,
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
        entity_embedding_weight: float = 1.0,
        mug_weight: float = 1.0,
        max_structural_entities: int = 1_000_000,
        scoring_function: str = "distmult",
        affine_p: int = 2,
        affine_aggr: str = "norm",
        affine_init_margin: float = 3.0,
        rotate_margin: float = 9.0,
        tucker_relation_dim: int | None = None,
        tucker_input_dropout: float = 0.0,
        tucker_relation_dropout: float = 0.0,
        tucker_hidden_dropout: float = 0.0,
        tucker_batch_norm: bool = True,
        pairre_margin: float = 9.0,
        pairre_p: int = 1,
        hybrid_weight: float = 1.0,
        quate_affine_weight: float = 0.0,
        quate_scale_weight: float = 0.0,
        quate_initializer: str = "uniform",
        use_entity_bias: bool = False,
        use_relation_entity_bias: bool = False,
        relation_entity_bias_init: str = "zeros",
        relation_entity_bias_init_scale: float = 0.1,
        relation_entity_bias_weight: float = 1.0,
    ):
        super().__init__()
        if scoring_function not in {
            "distmult",
            "affine",
            "complex",
            "tucker",
            "rotate",
            "quate",
            "pairre",
            "affine_distmult",
            "affine_complex",
        }:
            raise ValueError(
                "scoring_function must be 'distmult', 'affine', 'complex', 'tucker', "
                "'rotate', 'quate', 'pairre', 'affine_distmult', or 'affine_complex'."
            )
        if scoring_function in {"complex", "rotate", "affine_complex"} and embedding_dim % 2:
            raise ValueError(f"embedding_dim must be even for scoring_function={scoring_function!r}.")
        if scoring_function == "quate" and embedding_dim % 4:
            raise ValueError("embedding_dim must be divisible by 4 for scoring_function='quate'.")
        if pairre_p not in {1, 2}:
            raise ValueError("pairre_p must be 1 or 2.")
        self.multi_factory = multi_factory
        self.embedding_dim = embedding_dim
        self.entity_embedding_weight = entity_embedding_weight
        self.mug_weight = mug_weight
        self.scoring_function = scoring_function
        self.affine_p = affine_p
        if affine_aggr not in {"norm", "pow"}:
            raise ValueError("affine_aggr must be 'norm' or 'pow'.")
        self.affine_aggr = affine_aggr
        self.affine_init_margin = affine_init_margin
        self.rotate_margin = rotate_margin
        self.tucker_relation_dim = tucker_relation_dim or embedding_dim
        self.tucker_batch_norm = tucker_batch_norm
        self.pairre_margin = pairre_margin
        self.pairre_p = pairre_p
        self.hybrid_weight = hybrid_weight
        self.quate_affine_weight = quate_affine_weight
        self.quate_scale_weight = quate_scale_weight
        if quate_initializer not in {"uniform", "quaternion"}:
            raise ValueError("quate_initializer must be 'uniform' or 'quaternion'.")
        self.quate_initializer = quate_initializer
        self.use_entity_bias = use_entity_bias
        self.use_relation_entity_bias = use_relation_entity_bias
        self.relation_entity_bias_init = relation_entity_bias_init
        self.relation_entity_bias_init_scale = relation_entity_bias_init_scale
        self.relation_entity_bias_weight = relation_entity_bias_weight
        self.relation_embedding_dim = self.tucker_relation_dim if scoring_function == "tucker" else embedding_dim
        self.structural_dataset_names = {
            name
            for name in multi_factory.get_dataset_names()
            if multi_factory.get_num_entities(name) <= max_structural_entities
        }
        self.entity_representations = MUGEntityRepresentation(
            multi_factory=multi_factory,
            embedding_dim=embedding_dim,
            feature_mode=feature_mode,
            nb_max_length=nb_max_length,
            nb_top_k=nb_top_k,
            nb_min_count=nb_min_count,
            nb_max_two_hop_paths=nb_max_two_hop_paths,
            nb_max_two_hop_paths_per_middle=nb_max_two_hop_paths_per_middle,
            nb_max_edges_per_view=nb_max_edges_per_view,
            lambda_align=lambda_align,
            lambda_recon=lambda_recon,
            lambda_scatter=lambda_scatter,
            dropout=dropout,
            use_dimension_alignment=use_dimension_alignment,
            use_nb_encoder=use_nb_encoder,
            cache_refresh_interval=cache_refresh_interval,
            view_refresh_size=view_refresh_size,
            view_refresh_strategy=view_refresh_strategy,
            structural_dataset_names=self.structural_dataset_names,
        )
        self.name_to_key = {name: _safe_key(name) for name in multi_factory.get_dataset_names()}
        self.relation_embeddings = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(multi_factory.get_num_relations(name), self.relation_embedding_dim)
                for name in multi_factory.get_dataset_names()
            }
        )
        self.tucker_cores = nn.ParameterDict(
            (
                {
                    self.name_to_key[name]: nn.Parameter(
                        torch.empty(embedding_dim, self.tucker_relation_dim, embedding_dim)
                    )
                    for name in multi_factory.get_dataset_names()
                }
                if scoring_function == "tucker"
                else {}
            )
        )
        self.tucker_input_dropout = nn.Dropout(p=tucker_input_dropout)
        self.tucker_relation_dropout = nn.Dropout(p=tucker_relation_dropout)
        self.tucker_hidden_dropout = nn.Dropout(p=tucker_hidden_dropout)
        self.tucker_bn0 = nn.ModuleDict(
            (
                {
                    self.name_to_key[name]: nn.BatchNorm1d(embedding_dim)
                    for name in multi_factory.get_dataset_names()
                }
                if scoring_function == "tucker" and tucker_batch_norm
                else {}
            )
        )
        self.tucker_bn1 = nn.ModuleDict(
            (
                {
                    self.name_to_key[name]: nn.BatchNorm1d(embedding_dim)
                    for name in multi_factory.get_dataset_names()
                }
                if scoring_function == "tucker" and tucker_batch_norm
                else {}
            )
        )
        self.pairre_relation_tail_embeddings = nn.ModuleDict(
            (
                {
                    self.name_to_key[name]: nn.Embedding(multi_factory.get_num_relations(name), embedding_dim)
                    for name in multi_factory.get_dataset_names()
                }
                if scoring_function == "pairre"
                else {}
            )
        )
        self.quate_relation_scales = nn.ModuleDict(
            (
                {
                    self.name_to_key[name]: nn.Embedding(multi_factory.get_num_relations(name), embedding_dim)
                    for name in multi_factory.get_dataset_names()
                }
                if scoring_function == "quate"
                else {}
            )
        )
        # Shallow-kges inspired affine relation parameters:
        # score(h, r, t) = margin_r - || h * w_r + b_r - t ||_p
        self.affine_relation_scales = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(multi_factory.get_num_relations(name), embedding_dim)
                for name in multi_factory.get_dataset_names()
            }
        )
        self.affine_relation_biases = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(multi_factory.get_num_relations(name), embedding_dim)
                for name in multi_factory.get_dataset_names()
            }
        )
        self.affine_relation_margins = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(multi_factory.get_num_relations(name), 1)
                for name in multi_factory.get_dataset_names()
            }
        )
        self.entity_embeddings = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(
                    multi_factory.get_num_entities(name),
                    embedding_dim,
                    sparse=True,
                )
                for name in multi_factory.get_dataset_names()
            }
        )
        self.head_entity_biases = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(multi_factory.get_num_entities(name), 1)
                for name in multi_factory.get_dataset_names()
            }
        )
        self.tail_entity_biases = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(multi_factory.get_num_entities(name), 1)
                for name in multi_factory.get_dataset_names()
            }
        )
        # Relation-specific entity popularity terms. These are a lightweight
        # version of the relation/entity bias terms commonly used by strong
        # 1-vs-all KGE baselines: score(h,r,t) += b_head[r,h] + b_tail[r,t].
        # They help FB15k237/WN18RR where each relation has a strong domain and
        # range, while keeping the shallow affine scorer unchanged.
        self.relation_head_entity_biases = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(
                    multi_factory.get_num_relations(name) * multi_factory.get_num_entities(name),
                    1,
                    sparse=False,
                )
                for name in multi_factory.get_dataset_names()
            }
        )
        self.relation_tail_entity_biases = nn.ModuleDict(
            {
                self.name_to_key[name]: nn.Embedding(
                    multi_factory.get_num_relations(name) * multi_factory.get_num_entities(name),
                    1,
                    sparse=False,
                )
                for name in multi_factory.get_dataset_names()
            }
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_parameters(self) -> None:
        """Initialize trainable entity and relation embeddings."""
        entity_bound = math.sqrt(6.0 / self.embedding_dim)
        for embedding in self.entity_embeddings.values():
            if self.scoring_function == "tucker":
                nn.init.xavier_normal_(embedding.weight)
            elif self.scoring_function == "quate" and self.quate_initializer == "quaternion":
                self._init_quaternion_blocks_(embedding.weight)
            else:
                # Use a dimension-based initializer instead of nn.init.xavier_uniform_
                # on the full (num_entities, dim) table. The latter becomes extremely
                # small on large KGs and leaves scores close to zero for many epochs.
                nn.init.uniform_(embedding.weight, a=-entity_bound, b=entity_bound)
        relation_bound = math.sqrt(6.0 / self.relation_embedding_dim)
        for embedding in self.relation_embeddings.values():
            if self.scoring_function == "rotate":
                nn.init.uniform_(embedding.weight, a=-math.pi, b=math.pi)
            elif self.scoring_function == "tucker":
                nn.init.xavier_normal_(embedding.weight)
            elif self.scoring_function == "quate" and self.quate_initializer == "quaternion":
                self._init_quaternion_blocks_(embedding.weight)
            else:
                nn.init.uniform_(embedding.weight, a=-relation_bound, b=relation_bound)
        for core in self.tucker_cores.values():
            nn.init.uniform_(core, a=-1.0, b=1.0)
        pairre_relation_bound = math.sqrt(6.0 / self.embedding_dim)
        for embedding in self.pairre_relation_tail_embeddings.values():
            nn.init.uniform_(embedding.weight, a=-pairre_relation_bound, b=pairre_relation_bound)
        for embedding in self.quate_relation_scales.values():
            nn.init.zeros_(embedding.weight)
        for embedding in self.affine_relation_scales.values():
            nn.init.uniform_(embedding.weight, a=-1.0, b=1.0)
        for embedding in self.affine_relation_biases.values():
            nn.init.normal_(embedding.weight, std=1.0e-3)
        for embedding in self.affine_relation_margins.values():
            nn.init.constant_(embedding.weight, self.affine_init_margin)
        for embedding in self.head_entity_biases.values():
            nn.init.zeros_(embedding.weight)
        for embedding in self.tail_entity_biases.values():
            nn.init.zeros_(embedding.weight)
        for embedding in self.relation_head_entity_biases.values():
            nn.init.zeros_(embedding.weight)
        for embedding in self.relation_tail_entity_biases.values():
            nn.init.zeros_(embedding.weight)
        if self.relation_entity_bias_init == "counts":
            self._initialize_relation_entity_biases_from_counts()
        elif self.relation_entity_bias_init != "zeros":
            raise ValueError("relation_entity_bias_init must be 'zeros' or 'counts'.")

    @staticmethod
    def _init_quaternion_blocks_(weight: torch.FloatTensor) -> torch.FloatTensor:
        """Initialize block-layout quaternion embeddings like PyKEEN's QuatE initializer.

        The custom scorer stores quaternion components as contiguous blocks
        ``[real | i | j | k]``. PyKEEN's initializer conceptually creates
        ``(..., quaternion_dim, 4)`` values, so we generate that layout first
        and then convert it to the block layout used here.
        """
        if weight.shape[-1] % 4:
            raise ValueError("quaternion block initialization requires dimension divisible by 4.")
        num_rows, embedding_dim = weight.shape
        quaternion_dim = embedding_dim // 4
        # Match pykeen.nn.init.init_quaternions scale for a table of shape
        # (num_rows, quaternion_dim, 4): modulus ~ U[-s, s],
        # s = 1 / sqrt(2 * num_rows * quaternion_dim).
        s = 1.0 / math.sqrt(2.0 * num_rows * quaternion_dim)
        modulus = 2.0 * s * torch.rand(num_rows, quaternion_dim, device=weight.device) - s
        phase = 2.0 * math.pi * torch.rand(num_rows, quaternion_dim, device=weight.device)
        real = modulus * phase.cos()
        imag = torch.rand(num_rows, quaternion_dim, 3, device=weight.device)
        imag = torch.nn.functional.normalize(imag, p=2, dim=-1)
        imag = imag * (modulus * phase.sin()).unsqueeze(dim=-1)
        initialized = torch.cat(
            [real, imag[..., 0], imag[..., 1], imag[..., 2]],
            dim=-1,
        )
        with torch.no_grad():
            weight.copy_(initialized)
        return weight

    def _initialize_relation_entity_biases_from_counts(self) -> None:
        """Initialize relation/entity biases from log training frequencies."""
        with torch.no_grad():
            for dataset_name in self.multi_factory.get_dataset_names():
                key = self.name_to_key[dataset_name]
                num_entities = self.multi_factory.get_num_entities(dataset_name)
                triples = self.multi_factory.get_training_factory(dataset_name).mapped_triples.detach().cpu()
                head_counts = torch.zeros_like(self.relation_head_entity_biases[key].weight)
                tail_counts = torch.zeros_like(self.relation_tail_entity_biases[key].weight)
                for h, r, t in triples.tolist():
                    head_counts[int(r) * num_entities + int(h), 0] += 1.0
                    tail_counts[int(r) * num_entities + int(t), 0] += 1.0
                self.relation_head_entity_biases[key].weight.copy_(
                    self.relation_entity_bias_init_scale * torch.log1p(head_counts).to(
                        device=self.relation_head_entity_biases[key].weight.device
                    )
                )
                self.relation_tail_entity_biases[key].weight.copy_(
                    self.relation_entity_bias_init_scale * torch.log1p(tail_counts).to(
                        device=self.relation_tail_entity_biases[key].weight.device
                    )
                )

    def get_sparse_parameters(self, *, include_entity_embeddings: bool = True) -> list[nn.Parameter]:
        """Return parameters that can receive sparse gradients."""
        parameters: list[nn.Parameter] = []
        if include_entity_embeddings:
            parameters.extend(self.entity_embeddings.parameters())
        for embedding in self.relation_head_entity_biases.values():
            if embedding.sparse:
                parameters.extend(embedding.parameters())
        for embedding in self.relation_tail_entity_biases.values():
            if embedding.sparse:
                parameters.extend(embedding.parameters())
        return parameters

    def clear_cache(self) -> None:
        """Clear cached entity representations."""
        self.entity_representations.clear_cache()

    def advance_cache_version(self) -> None:
        """Advance the structural cache version after a parameter update."""
        self.entity_representations.advance_cache_version()

    def get_all_entity_representations(self, dataset_name: str) -> torch.FloatTensor:
        """Return all entity representations for one dataset."""
        learned = self.entity_embeddings[self.name_to_key[dataset_name]].weight
        if dataset_name not in self.structural_dataset_names:
            return self.entity_embedding_weight * learned
        mug = self.entity_representations.get_all_entity_representations(dataset_name)
        return self.mug_weight * mug + self.entity_embedding_weight * learned

    def get_auxiliary_loss(self, dataset_name: str) -> torch.FloatTensor:
        """Return the auxiliary AMBRE loss for one dataset."""
        if dataset_name not in self.structural_dataset_names:
            return next(self.parameters()).new_zeros(())
        return self.entity_representations.get_auxiliary_loss(dataset_name)

    def _relation(self, dataset_name: str, relation_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.relation_embeddings[self.name_to_key[dataset_name]](relation_ids.to(device=self.device))

    def _entity(self, dataset_name: str, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        entity_ids = entity_ids.to(device=self.device)
        learned = self.entity_embeddings[self.name_to_key[dataset_name]](entity_ids)
        if dataset_name not in self.structural_dataset_names:
            return self.entity_embedding_weight * learned
        mug = self.entity_representations(dataset_name, entity_ids)
        return self.mug_weight * mug + self.entity_embedding_weight * learned

    def _head_bias(self, dataset_name: str, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.head_entity_biases[self.name_to_key[dataset_name]](entity_ids.to(device=self.device)).squeeze(dim=-1)

    def _tail_bias(self, dataset_name: str, entity_ids: torch.LongTensor) -> torch.FloatTensor:
        return self.tail_entity_biases[self.name_to_key[dataset_name]](entity_ids.to(device=self.device)).squeeze(dim=-1)

    def _all_head_biases(self, dataset_name: str) -> torch.FloatTensor:
        return self.head_entity_biases[self.name_to_key[dataset_name]].weight.squeeze(dim=-1)

    def _all_tail_biases(self, dataset_name: str) -> torch.FloatTensor:
        return self.tail_entity_biases[self.name_to_key[dataset_name]].weight.squeeze(dim=-1)

    def _relation_entity_index(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
        entity_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        num_entities = self.multi_factory.get_num_entities(dataset_name)
        return relation_ids.to(device=self.device) * num_entities + entity_ids.to(device=self.device)

    def _relation_head_entity_bias(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
        entity_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        key = self.name_to_key[dataset_name]
        num_entities = self.multi_factory.get_num_entities(dataset_name)
        matrix = self.relation_head_entity_biases[key].weight.squeeze(dim=-1).view(-1, num_entities)
        return self.relation_entity_bias_weight * matrix[
            relation_ids.to(device=self.device),
            entity_ids.to(device=self.device),
        ]

    def _relation_tail_entity_bias(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
        entity_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        key = self.name_to_key[dataset_name]
        num_entities = self.multi_factory.get_num_entities(dataset_name)
        matrix = self.relation_tail_entity_biases[key].weight.squeeze(dim=-1).view(-1, num_entities)
        return self.relation_entity_bias_weight * matrix[
            relation_ids.to(device=self.device),
            entity_ids.to(device=self.device),
        ]

    def _all_relation_head_entity_biases(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        key = self.name_to_key[dataset_name]
        num_entities = self.multi_factory.get_num_entities(dataset_name)
        matrix = self.relation_head_entity_biases[key].weight.squeeze(dim=-1).view(-1, num_entities)
        return self.relation_entity_bias_weight * matrix.index_select(dim=0, index=relation_ids.to(device=self.device))

    def _all_relation_tail_entity_biases(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        key = self.name_to_key[dataset_name]
        num_entities = self.multi_factory.get_num_entities(dataset_name)
        matrix = self.relation_tail_entity_biases[key].weight.squeeze(dim=-1).view(-1, num_entities)
        return self.relation_entity_bias_weight * matrix.index_select(dim=0, index=relation_ids.to(device=self.device))

    def _add_biases(
        self,
        dataset_name: str,
        scores: torch.FloatTensor,
        head_ids: torch.LongTensor,
        relation_ids: torch.LongTensor,
        tail_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        if self.use_entity_bias:
            scores = scores + self._head_bias(dataset_name, head_ids) + self._tail_bias(dataset_name, tail_ids)
        if self.use_relation_entity_bias:
            scores = scores + self._relation_head_entity_bias(dataset_name, relation_ids, head_ids)
            scores = scores + self._relation_tail_entity_bias(dataset_name, relation_ids, tail_ids)
        return scores

    def _affine_relation(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        key = self.name_to_key[dataset_name]
        relation_ids = relation_ids.to(device=self.device)
        scale = self.affine_relation_scales[key](relation_ids)
        bias = self.affine_relation_biases[key](relation_ids)
        margin = self.affine_relation_margins[key](relation_ids).squeeze(dim=-1)
        return scale, bias, margin

    @staticmethod
    def _split_complex(x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return x.tensor_split(2, dim=-1)

    @staticmethod
    def _complex_score(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        h_re, h_im = MUGKGC._split_complex(h)
        r_re, r_im = MUGKGC._split_complex(r)
        t_re, t_im = MUGKGC._split_complex(t)
        return (
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

    def _rotate_relation(self, r: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        phase = r[..., : self.embedding_dim // 2]
        return torch.cos(phase), torch.sin(phase)

    @staticmethod
    def _complex_modulus_l1_scores(
        query_re: torch.FloatTensor,
        query_im: torch.FloatTensor,
        candidates_re: torch.FloatTensor,
        candidates_im: torch.FloatTensor,
        margin: float,
        eps: float = 1.0e-12,
    ) -> torch.FloatTensor:
        batch_size, dim = query_re.shape
        num_candidates = candidates_re.shape[0]
        # Vectorize over rows. Chunk over candidates to keep the temporary
        # [batch, candidates, dim] tensor bounded during LCWA/evaluation.
        max_elements = 80_000_000
        chunk_size = max(1, min(num_candidates, max_elements // max(1, batch_size * dim)))
        chunks = []
        for start in range(0, num_candidates, chunk_size):
            stop = min(start + chunk_size, num_candidates)
            diff_re = query_re.unsqueeze(dim=1) - candidates_re[start:stop].unsqueeze(dim=0)
            diff_im = query_im.unsqueeze(dim=1) - candidates_im[start:stop].unsqueeze(dim=0)
            distance = (diff_re.square() + diff_im.square() + eps).sqrt().sum(dim=-1)
            chunks.append(margin - distance)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def _split_quaternion(
        x: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return x.tensor_split(4, dim=-1)

    @staticmethod
    def _quate_unit_relation(
        r: torch.FloatTensor,
        eps: float = 1.0e-12,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        r_a, r_b, r_c, r_d = MUGKGC._split_quaternion(r)
        norm = (r_a.square() + r_b.square() + r_c.square() + r_d.square() + eps).sqrt()
        return r_a / norm, r_b / norm, r_c / norm, r_d / norm

    @staticmethod
    def _quate_rotate(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        h_a, h_b, h_c, h_d = MUGKGC._split_quaternion(h)
        p, q, u, v = MUGKGC._quate_unit_relation(r)
        return (
            h_a * p - h_b * q - h_c * u - h_d * v,
            h_a * q + h_b * p + h_c * v - h_d * u,
            h_a * u - h_b * v + h_c * p + h_d * q,
            h_a * v + h_b * u - h_c * q + h_d * p,
        )

    @staticmethod
    def _quate_score(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        rot_a, rot_b, rot_c, rot_d = MUGKGC._quate_rotate(h=h, r=r)
        t_a, t_b, t_c, t_d = MUGKGC._split_quaternion(t)
        return (rot_a * t_a + rot_b * t_b + rot_c * t_c + rot_d * t_d).sum(dim=-1)

    def _quate_relation_scale(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor] | None:
        """Return optional relation-wise quaternion dimension gates."""
        if not self.quate_scale_weight:
            return None
        scale = self.quate_relation_scales[self.name_to_key[dataset_name]](
            relation_ids.to(device=self.device)
        )
        scale = 1.0 + self.quate_scale_weight * scale
        return self._split_quaternion(scale)

    def _tucker_relation_matrix(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        relation = self._relation(dataset_name, relation_ids)
        core = self.tucker_cores[self.name_to_key[dataset_name]]
        # Match PyKEEN's TuckERInteraction:
        #   relation_matrix[b, i, k] = sum_j core[i, j, k] * relation[b, j]
        relation_matrix = torch.einsum("ijk,bj->bik", core, relation)
        return self.tucker_relation_dropout(relation_matrix)

    def _tucker_prepare_input(self, dataset_name: str, h: torch.FloatTensor) -> torch.FloatTensor:
        if self.tucker_batch_norm:
            h = self.tucker_bn0[self.name_to_key[dataset_name]](h)
        return self.tucker_input_dropout(h)

    def _tucker_prepare_hidden(self, dataset_name: str, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.tucker_batch_norm:
            x = self.tucker_bn1[self.name_to_key[dataset_name]](x)
        return self.tucker_hidden_dropout(x)

    def _tucker_project(
        self,
        dataset_name: str,
        h: torch.FloatTensor,
        relation_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        h = self._tucker_prepare_input(dataset_name, h)
        relation_matrix = self._tucker_relation_matrix(dataset_name, relation_ids)
        x = torch.bmm(h.unsqueeze(dim=1), relation_matrix).squeeze(dim=1)
        return self._tucker_prepare_hidden(dataset_name, x)

    def _tucker_score_h(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
        t: torch.FloatTensor,
        z: torch.FloatTensor,
    ) -> torch.FloatTensor:
        relation_matrix = self._tucker_relation_matrix(dataset_name, relation_ids)
        if not self.tucker_batch_norm:
            query = torch.bmm(relation_matrix, t.unsqueeze(dim=-1)).squeeze(dim=-1)
            return query @ z.t()
        candidates = self._tucker_prepare_input(dataset_name, z)
        rows = []
        for row in range(relation_ids.shape[0]):
            x = candidates @ relation_matrix[row]
            x = self._tucker_prepare_hidden(dataset_name, x)
            rows.append(x @ t[row])
        return torch.stack(rows, dim=0)

    def _pairre_relation(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        key = self.name_to_key[dataset_name]
        relation_ids = relation_ids.to(device=self.device)
        return self.relation_embeddings[key](relation_ids), self.pairre_relation_tail_embeddings[key](relation_ids)

    def _pairre_score_hrt(
        self,
        dataset_name: str,
        h: torch.FloatTensor,
        relation_ids: torch.LongTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r_h, r_t = self._pairre_relation(dataset_name, relation_ids)
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
        t = torch.nn.functional.normalize(t, p=2, dim=-1)
        distance = torch.linalg.vector_norm(h * r_h - t * r_t, ord=self.pairre_p, dim=-1)
        return self.pairre_margin - distance

    def _pairre_score_t(
        self,
        dataset_name: str,
        h: torch.FloatTensor,
        relation_ids: torch.LongTensor,
        z: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r_h, r_t = self._pairre_relation(dataset_name, relation_ids)
        h = torch.nn.functional.normalize(h, p=2, dim=-1)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        query = h * r_h
        return self._pairre_chunked_scores(
            query=query,
            candidates=z,
            candidate_relation=r_t,
        )

    def _pairre_score_h(
        self,
        dataset_name: str,
        relation_ids: torch.LongTensor,
        t: torch.FloatTensor,
        z: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r_h, r_t = self._pairre_relation(dataset_name, relation_ids)
        t = torch.nn.functional.normalize(t, p=2, dim=-1)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        query = t * r_t
        return self._pairre_chunked_scores(
            query=query,
            candidates=z,
            candidate_relation=r_h,
        )

    def _pairre_chunked_scores(
        self,
        *,
        query: torch.FloatTensor,
        candidates: torch.FloatTensor,
        candidate_relation: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Score PairRE queries against all candidates without a Python row loop."""
        batch_size, dim = query.shape
        num_candidates = candidates.shape[0]
        max_elements = 80_000_000
        chunk_size = max(1, min(num_candidates, max_elements // max(1, batch_size * dim)))
        chunks = []
        for start in range(0, num_candidates, chunk_size):
            stop = min(start + chunk_size, num_candidates)
            transformed_candidates = candidates[start:stop].unsqueeze(dim=0) * candidate_relation.unsqueeze(dim=1)
            diff = transformed_candidates - query.unsqueeze(dim=1)
            if self.pairre_p == 1:
                distance = diff.abs().sum(dim=-1)
            else:
                distance = diff.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
            chunks.append(self.pairre_margin - distance)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def _pairwise_l2_scores(
        query: torch.FloatTensor,
        candidates: torch.FloatTensor,
        margin: torch.FloatTensor,
    ) -> torch.FloatTensor:
        query_sq = query.pow(2).sum(dim=-1, keepdim=True)
        candidate_sq = candidates.pow(2).sum(dim=-1).unsqueeze(dim=0)
        distances_sq = (query_sq + candidate_sq - 2.0 * query @ candidates.t()).clamp_min(1.0e-12)
        return margin.unsqueeze(dim=-1) - distances_sq.sqrt()

    def _affine_score_hrt(
        self,
        dataset_name: str,
        h: torch.FloatTensor,
        relation_ids: torch.LongTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        scale, bias, margin = self._affine_relation(dataset_name, relation_ids)
        transformed = h * scale + bias
        difference = transformed - t
        if self.affine_aggr == "pow":
            return margin - difference.abs().pow(self.affine_p).sum(dim=-1)
        return margin - torch.linalg.vector_norm(difference, ord=self.affine_p, dim=-1)

    def _affine_score_t(self, dataset_name: str, h: torch.FloatTensor, relation_ids: torch.LongTensor, z: torch.FloatTensor) -> torch.FloatTensor:
        scale, bias, margin = self._affine_relation(dataset_name, relation_ids)
        query = h * scale + bias
        if self.affine_aggr == "pow":
            query_sq = query.pow(2).sum(dim=-1, keepdim=True)
            candidate_sq = z.pow(2).sum(dim=-1).unsqueeze(dim=0)
            distances_sq = (query_sq + candidate_sq - 2.0 * query @ z.t()).clamp_min(0.0)
            return margin.unsqueeze(dim=-1) - distances_sq
        if self.affine_p == 2:
            return self._pairwise_l2_scores(query=query, candidates=z, margin=margin)
        return margin.unsqueeze(dim=-1) - torch.cdist(query, z, p=self.affine_p)

    def _affine_score_h(self, dataset_name: str, relation_ids: torch.LongTensor, t: torch.FloatTensor, z: torch.FloatTensor) -> torch.FloatTensor:
        scale, bias, margin = self._affine_relation(dataset_name, relation_ids)
        rows = []
        for row in range(relation_ids.shape[0]):
            candidates = z * scale[row].unsqueeze(dim=0) + bias[row].unsqueeze(dim=0)
            if self.affine_aggr == "pow":
                rows.append(margin[row] - (candidates - t[row].unsqueeze(dim=0)).abs().pow(self.affine_p).sum(dim=-1))
                continue
            if self.affine_p == 2:
                rows.append(
                    self._pairwise_l2_scores(
                        query=t[row : row + 1],
                        candidates=candidates,
                        margin=margin[row : row + 1],
                    ).squeeze(dim=0)
                )
            else:
                rows.append(margin[row] - torch.linalg.vector_norm(candidates - t[row].unsqueeze(dim=0), ord=self.affine_p, dim=-1))
        return torch.stack(rows, dim=0)

    def score_hrt(self, dataset_name: str, triples: torch.LongTensor) -> torch.FloatTensor:
        """Score triples from one dataset."""
        triples = triples.to(device=self.device)
        h = self._entity(dataset_name, triples[:, 0])
        t = self._entity(dataset_name, triples[:, 2])
        if self.scoring_function == "distmult":
            r = self._relation(dataset_name, triples[:, 1])
            scores = (h * r * t).sum(dim=-1)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "complex":
            r = self._relation(dataset_name, triples[:, 1])
            scores = self._complex_score(h=h, r=r, t=t)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "tucker":
            x = self._tucker_project(dataset_name, h, triples[:, 1])
            scores = (x * t).sum(dim=-1)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "rotate":
            r = self._relation(dataset_name, triples[:, 1])
            h_re, h_im = self._split_complex(h)
            t_re, t_im = self._split_complex(t)
            r_re, r_im = self._rotate_relation(r)
            query_re = h_re * r_re - h_im * r_im
            query_im = h_re * r_im + h_im * r_re
            scores = self.rotate_margin - ((query_re - t_re).square() + (query_im - t_im).square() + 1.0e-12).sqrt().sum(dim=-1)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "quate":
            r = self._relation(dataset_name, triples[:, 1])
            rot_a, rot_b, rot_c, rot_d = self._quate_rotate(h=h, r=r)
            scale = self._quate_relation_scale(dataset_name, triples[:, 1])
            if scale is not None:
                scale_a, scale_b, scale_c, scale_d = scale
                rot_a = rot_a * scale_a
                rot_b = rot_b * scale_b
                rot_c = rot_c * scale_c
                rot_d = rot_d * scale_d
            t_a, t_b, t_c, t_d = self._split_quaternion(t)
            scores = (rot_a * t_a + rot_b * t_b + rot_c * t_c + rot_d * t_d).sum(dim=-1)
            if self.quate_affine_weight:
                scores = scores + self.quate_affine_weight * self._affine_score_hrt(
                    dataset_name,
                    h,
                    triples[:, 1],
                    t,
                )
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "pairre":
            scores = self._pairre_score_hrt(dataset_name, h, triples[:, 1], t)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "affine_distmult":
            r = self._relation(dataset_name, triples[:, 1])
            scores = self._affine_score_hrt(dataset_name, h, triples[:, 1], t) + self.hybrid_weight * (h * r * t).sum(dim=-1)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])
        if self.scoring_function == "affine_complex":
            r = self._relation(dataset_name, triples[:, 1])
            scores = self._affine_score_hrt(dataset_name, h, triples[:, 1], t) + self.hybrid_weight * self._complex_score(h=h, r=r, t=t)
            return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])

        scores = self._affine_score_hrt(dataset_name, h, triples[:, 1], t)
        return self._add_biases(dataset_name, scores, triples[:, 0], triples[:, 1], triples[:, 2])

    def score_t(self, dataset_name: str, hr_batch: torch.LongTensor) -> torch.FloatTensor:
        """Score all candidate tails from the same dataset for each ``(h, r)`` pair."""
        hr_batch = hr_batch.to(device=self.device)
        z = self.get_all_entity_representations(dataset_name)
        h = z[hr_batch[:, 0]]
        if self.scoring_function == "distmult":
            r = self._relation(dataset_name, hr_batch[:, 1])
            query = h * r
            scores = query @ z.t()
            if self.use_entity_bias:
                scores = scores + self._tail_bias(dataset_name, hr_batch[:, 0]).unsqueeze(dim=-1) * 0.0 + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "complex":
            r = self._relation(dataset_name, hr_batch[:, 1])
            h_re, h_im = self._split_complex(h)
            r_re, r_im = self._split_complex(r)
            z_re, z_im = self._split_complex(z)
            query_re = h_re * r_re - h_im * r_im
            query_im = h_re * r_im + h_im * r_re
            scores = query_re @ z_re.t() + query_im @ z_im.t()
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "tucker":
            x = self._tucker_project(dataset_name, h, hr_batch[:, 1])
            scores = x @ z.t()
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "rotate":
            r = self._relation(dataset_name, hr_batch[:, 1])
            h_re, h_im = self._split_complex(h)
            z_re, z_im = self._split_complex(z)
            r_re, r_im = self._rotate_relation(r)
            query_re = h_re * r_re - h_im * r_im
            query_im = h_re * r_im + h_im * r_re
            scores = self._complex_modulus_l1_scores(
                query_re=query_re,
                query_im=query_im,
                candidates_re=z_re,
                candidates_im=z_im,
                margin=self.rotate_margin,
            )
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "quate":
            r = self._relation(dataset_name, hr_batch[:, 1])
            rot_a, rot_b, rot_c, rot_d = self._quate_rotate(h=h, r=r)
            scale = self._quate_relation_scale(dataset_name, hr_batch[:, 1])
            if scale is not None:
                scale_a, scale_b, scale_c, scale_d = scale
                rot_a = rot_a * scale_a
                rot_b = rot_b * scale_b
                rot_c = rot_c * scale_c
                rot_d = rot_d * scale_d
            z_a, z_b, z_c, z_d = self._split_quaternion(z)
            scores = rot_a @ z_a.t() + rot_b @ z_b.t() + rot_c @ z_c.t() + rot_d @ z_d.t()
            if self.quate_affine_weight:
                scores = scores + self.quate_affine_weight * self._affine_score_t(
                    dataset_name,
                    h,
                    hr_batch[:, 1],
                    z,
                )
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "pairre":
            scores = self._pairre_score_t(dataset_name, h, hr_batch[:, 1], z)
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "affine_distmult":
            r = self._relation(dataset_name, hr_batch[:, 1])
            query = h * r
            scores = self._affine_score_t(dataset_name, h, hr_batch[:, 1], z) + self.hybrid_weight * (query @ z.t())
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores
        if self.scoring_function == "affine_complex":
            r = self._relation(dataset_name, hr_batch[:, 1])
            h_re, h_im = self._split_complex(h)
            r_re, r_im = self._split_complex(r)
            z_re, z_im = self._split_complex(z)
            query_re = h_re * r_re - h_im * r_im
            query_im = h_re * r_im + h_im * r_re
            scores = self._affine_score_t(dataset_name, h, hr_batch[:, 1], z) + self.hybrid_weight * (
                query_re @ z_re.t() + query_im @ z_im.t()
            )
            if self.use_entity_bias:
                scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
                scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
            return scores

        scores = self._affine_score_t(dataset_name, h, hr_batch[:, 1], z)
        if self.use_entity_bias:
            scores = scores + self._all_tail_biases(dataset_name).unsqueeze(dim=0)
        if self.use_relation_entity_bias:
            scores = scores + self._relation_head_entity_bias(dataset_name, hr_batch[:, 1], hr_batch[:, 0]).unsqueeze(dim=-1)
            scores = scores + self._all_relation_tail_entity_biases(dataset_name, hr_batch[:, 1])
        return scores

    def score_h(self, dataset_name: str, rt_batch: torch.LongTensor) -> torch.FloatTensor:
        """Score all candidate heads from the same dataset for each ``(r, t)`` pair."""
        rt_batch = rt_batch.to(device=self.device)
        z = self.get_all_entity_representations(dataset_name)
        t = z[rt_batch[:, 1]]
        if self.scoring_function == "distmult":
            r = self._relation(dataset_name, rt_batch[:, 0])
            query = r * t
            scores = query @ z.t()
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "complex":
            r = self._relation(dataset_name, rt_batch[:, 0])
            r_re, r_im = self._split_complex(r)
            t_re, t_im = self._split_complex(t)
            z_re, z_im = self._split_complex(z)
            query_re = r_re * t_re + r_im * t_im
            query_im = r_re * t_im - r_im * t_re
            scores = query_re @ z_re.t() + query_im @ z_im.t()
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "tucker":
            scores = self._tucker_score_h(dataset_name, rt_batch[:, 0], t, z)
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "rotate":
            r = self._relation(dataset_name, rt_batch[:, 0])
            t_re, t_im = self._split_complex(t)
            z_re, z_im = self._split_complex(z)
            r_re, r_im = self._rotate_relation(r)
            query_re = t_re * r_re + t_im * r_im
            query_im = -t_re * r_im + t_im * r_re
            scores = self._complex_modulus_l1_scores(
                query_re=query_re,
                query_im=query_im,
                candidates_re=z_re,
                candidates_im=z_im,
                margin=self.rotate_margin,
            )
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "quate":
            r = self._relation(dataset_name, rt_batch[:, 0])
            p, q, u, v = self._quate_unit_relation(r)
            t_a, t_b, t_c, t_d = self._split_quaternion(t)
            scale = self._quate_relation_scale(dataset_name, rt_batch[:, 0])
            if scale is not None:
                scale_a, scale_b, scale_c, scale_d = scale
                t_a = t_a * scale_a
                t_b = t_b * scale_b
                t_c = t_c * scale_c
                t_d = t_d * scale_d
            z_a, z_b, z_c, z_d = self._split_quaternion(z)
            query_a = p * t_a + q * t_b + u * t_c + v * t_d
            query_b = -q * t_a + p * t_b - v * t_c + u * t_d
            query_c = -u * t_a + v * t_b + p * t_c - q * t_d
            query_d = -v * t_a - u * t_b + q * t_c + p * t_d
            scores = query_a @ z_a.t() + query_b @ z_b.t() + query_c @ z_c.t() + query_d @ z_d.t()
            if self.quate_affine_weight:
                scores = scores + self.quate_affine_weight * self._affine_score_h(
                    dataset_name,
                    rt_batch[:, 0],
                    t,
                    z,
                )
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "pairre":
            scores = self._pairre_score_h(dataset_name, rt_batch[:, 0], t, z)
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "affine_distmult":
            r = self._relation(dataset_name, rt_batch[:, 0])
            query = r * t
            scores = self._affine_score_h(dataset_name, rt_batch[:, 0], t, z) + self.hybrid_weight * (query @ z.t())
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores
        if self.scoring_function == "affine_complex":
            r = self._relation(dataset_name, rt_batch[:, 0])
            r_re, r_im = self._split_complex(r)
            t_re, t_im = self._split_complex(t)
            z_re, z_im = self._split_complex(z)
            query_re = r_re * t_re + r_im * t_im
            query_im = r_re * t_im - r_im * t_re
            scores = self._affine_score_h(dataset_name, rt_batch[:, 0], t, z) + self.hybrid_weight * (
                query_re @ z_re.t() + query_im @ z_im.t()
            )
            if self.use_entity_bias:
                scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
            if self.use_relation_entity_bias:
                scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
                scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
            return scores

        scores = self._affine_score_h(dataset_name, rt_batch[:, 0], t, z)
        if self.use_entity_bias:
            scores = scores + self._all_head_biases(dataset_name).unsqueeze(dim=0)
        if self.use_relation_entity_bias:
            scores = scores + self._all_relation_head_entity_biases(dataset_name, rt_batch[:, 0])
            scores = scores + self._relation_tail_entity_bias(dataset_name, rt_batch[:, 0], rt_batch[:, 1]).unsqueeze(dim=-1)
        return scores
