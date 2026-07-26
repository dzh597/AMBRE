"""AMBRE-style alignment and shared non-backtracking spectral encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DimensionAwareAlignment(nn.Module):
    """Project variable-dimensional structural features into one shared space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.basis)

    def forward(self, x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return aligned features and the alignment regularization loss."""
        aligned = x @ self.basis
        align_loss = self.basis.mean(dim=0).pow(2).sum()
        return aligned, align_loss


class StandardFeatureProjection(nn.Module):
    """A plain per-dataset linear projection for the no-alignment ablation."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return projected features and a zero alignment loss."""
        return self.linear(x), x.new_zeros(())


def _build_edge_index(adjacency: torch.Tensor) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Return CPU edge-source and edge-target index vectors from a sparse adjacency."""
    adjacency = adjacency.coalesce()
    indices = adjacency.indices().detach().cpu()
    if indices.numel() == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    return indices[0].clone(), indices[1].clone()


class SharedNonBacktrackingSpectralEncoder(nn.Module):
    """Shared non-backtracking spectral encoder over relation-sequence views."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_reconstruction_edges: int = 2048,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_reconstruction_edges = max_reconstruction_edges

        self.theta_src = nn.Linear(input_dim, hidden_dim, bias=False)
        self.theta_dst = nn.Linear(input_dim, hidden_dim, bias=False)
        self.theta_nb = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.beta = nn.Parameter(torch.zeros(num_layers + 1))
        self.beta.data[0] = 1.0

    @staticmethod
    def _activation(x: torch.FloatTensor) -> torch.FloatTensor:
        return F.relu(x)

    @staticmethod
    def _build_node_edge_lists(
        src: torch.LongTensor,
        dst: torch.LongTensor,
        num_nodes: int,
    ) -> tuple[list[list[int]], list[list[int]], dict[tuple[int, int], int]]:
        """Build incoming/outgoing edge lists and an oriented-edge lookup table."""
        incoming: list[list[int]] = [[] for _ in range(num_nodes)]
        outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
        lookup: dict[tuple[int, int], int] = {}
        for edge_id, (u, v) in enumerate(zip(src.tolist(), dst.tolist(), strict=False)):
            incoming[v].append(edge_id)
            outgoing[u].append(edge_id)
            lookup[(u, v)] = edge_id
        return incoming, outgoing, lookup

    @staticmethod
    def _build_edge_lookup(
        src: torch.LongTensor,
        dst: torch.LongTensor,
    ) -> dict[tuple[int, int], int]:
        """Build an oriented-edge lookup table."""
        return {
            (int(u), int(v)): edge_id
            for edge_id, (u, v) in enumerate(zip(src.tolist(), dst.tolist(), strict=False))
        }

    @staticmethod
    def _build_reverse_edge_index(
        src: torch.LongTensor,
        dst: torch.LongTensor,
        lookup: dict[tuple[int, int], int],
        *,
        device: torch.device,
    ) -> torch.LongTensor:
        """Return the reverse edge id for every oriented edge, or -1 if absent."""
        reverse_index = torch.full((src.shape[0],), -1, dtype=torch.long, device=device)
        for edge_id, (u, v) in enumerate(zip(src.tolist(), dst.tolist(), strict=False)):
            reverse_index[edge_id] = lookup.get((int(v), int(u)), -1)
        return reverse_index

    def _non_backtracking_step(
        self,
        previous: torch.FloatTensor,
        src: torch.LongTensor,
        dst: torch.LongTensor,
        incoming: list[list[int]],
        outgoing: list[list[int]],
        lookup: dict[tuple[int, int], int],
    ) -> torch.FloatTensor:
        """Apply one normalized non-backtracking propagation step."""
        if previous.shape[0] == 0:
            return previous.new_zeros(previous.shape)
        reverse_index = self._build_reverse_edge_index(
            src=src,
            dst=dst,
            lookup=lookup,
            device=previous.device,
        )
        return self._non_backtracking_step_with_reverse_index(
            previous=previous,
            src=src.to(device=previous.device),
            dst=dst.to(device=previous.device),
            reverse_index=reverse_index,
            num_nodes=len(incoming),
        )

    def _non_backtracking_step_with_reverse_index(
        self,
        previous: torch.FloatTensor,
        src: torch.LongTensor,
        dst: torch.LongTensor,
        reverse_index: torch.LongTensor,
        *,
        num_nodes: int,
    ) -> torch.FloatTensor:
        """Apply one normalized non-backtracking propagation step with cached reverse edges."""
        next_state = previous.new_zeros(previous.shape)
        incoming_sum = previous.new_zeros((num_nodes, previous.shape[1]))
        incoming_sum.index_add_(0, dst, previous)
        incoming_count = torch.bincount(dst, minlength=num_nodes)

        edge_sum = incoming_sum.index_select(0, src)
        edge_count = incoming_count.index_select(0, src).to(device=previous.device, dtype=previous.dtype)

        reverse_mask = reverse_index >= 0
        if reverse_mask.any():
            edge_sum = edge_sum.clone()
            edge_count = edge_count.clone()
            edge_sum[reverse_mask] = edge_sum[reverse_mask] - previous.index_select(0, reverse_index[reverse_mask])
            edge_count[reverse_mask] = edge_count[reverse_mask] - 1.0

        valid = edge_count > 0
        if valid.any():
            next_state[valid] = edge_sum[valid] / edge_count[valid].unsqueeze(dim=-1)
        return next_state

    def _edge_reconstruction_loss(
        self,
        z: torch.FloatTensor,
        adjacency: torch.Tensor,
    ) -> torch.FloatTensor:
        """Sample positive/negative pairs from one view and reconstruct its adjacency."""
        adjacency = adjacency.coalesce().cpu()
        src, dst = _build_edge_index(adjacency)
        num_edges = src.shape[0]
        if num_edges == 0:
            return z.new_zeros(())

        if num_edges > self.max_reconstruction_edges:
            choice = torch.randperm(num_edges)[: self.max_reconstruction_edges]
            src = src.index_select(0, choice)
            dst = dst.index_select(0, choice)
            num_edges = src.shape[0]

        device = z.device
        pos_h = src.to(device=device)
        pos_t = dst.to(device=device)
        pos_scores = (z[pos_h] * z[pos_t]).sum(dim=-1)

        edge_set = set(zip(src.tolist(), dst.tolist(), strict=False))
        neg_pairs: list[tuple[int, int]] = []
        attempts = 0
        max_attempts = max(self.max_reconstruction_edges * 20, 1)
        while len(neg_pairs) < num_edges and attempts < max_attempts:
            remaining = num_edges - len(neg_pairs)
            sample_count = max(remaining * 2, 16)
            heads = torch.randint(high=z.shape[0], size=(sample_count,), device=device)
            tails = torch.randint(high=z.shape[0], size=(sample_count,), device=device)
            for h, t in zip(heads.tolist(), tails.tolist(), strict=False):
                if (h, t) in edge_set:
                    continue
                neg_pairs.append((h, t))
                if len(neg_pairs) >= num_edges:
                    break
            attempts += sample_count

        if not neg_pairs:
            labels = torch.ones_like(pos_scores)
            return F.binary_cross_entropy_with_logits(pos_scores, labels)

        neg = torch.as_tensor(neg_pairs, dtype=torch.long, device=device)
        neg_scores = (z[neg[:, 0]] * z[neg[:, 1]]).sum(dim=-1)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def _encode_one(
        self,
        adjacency: torch.Tensor,
        x: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Encode one relation-sequence view and return view-specific losses."""
        adjacency = adjacency.coalesce()
        src, dst = _build_edge_index(adjacency)
        num_nodes = adjacency.shape[0]

        h0 = x
        if src.numel() == 0:
            z = self._activation(h0)
            return z, self._edge_reconstruction_loss(z=z, adjacency=adjacency)

        lookup = self._build_edge_lookup(src=src, dst=dst)
        src_device = src.to(device=x.device)
        dst_device = dst.to(device=x.device)
        reverse_index = self._build_reverse_edge_index(src=src, dst=dst, lookup=lookup, device=x.device)

        edge_state = self.theta_src(h0.index_select(0, src_device)) + self.theta_dst(h0.index_select(0, dst_device))
        states = [edge_state]
        for _ in range(self.num_layers):
            states.append(
                self._non_backtracking_step_with_reverse_index(
                    previous=states[-1],
                    src=src_device,
                    dst=dst_device,
                    reverse_index=reverse_index,
                    num_nodes=num_nodes,
                )
            )

        filtered = torch.zeros_like(states[0])
        for order, state in enumerate(states):
            filtered = filtered + self.beta[order] * state
        if self.dropout > 0.0:
            filtered = F.dropout(filtered, p=self.dropout, training=self.training)
        edge_repr = self._activation(self.theta_nb(filtered))

        incoming_summary = edge_repr.new_zeros((num_nodes, self.hidden_dim))
        outgoing_summary = edge_repr.new_zeros((num_nodes, self.hidden_dim))
        incoming_sum = incoming_summary.clone()
        outgoing_sum = outgoing_summary.clone()
        incoming_sum.index_add_(0, dst_device, edge_repr)
        outgoing_sum.index_add_(0, src_device, edge_repr)
        incoming_count = torch.bincount(dst_device, minlength=num_nodes)
        outgoing_count = torch.bincount(src_device, minlength=num_nodes)
        incoming_mask = incoming_count > 0
        outgoing_mask = outgoing_count > 0
        if incoming_mask.any():
            incoming_summary[incoming_mask] = incoming_sum[incoming_mask] / incoming_count[incoming_mask].to(
                device=edge_repr.device,
                dtype=edge_repr.dtype,
            ).unsqueeze(dim=-1)
        if outgoing_mask.any():
            outgoing_summary[outgoing_mask] = outgoing_sum[outgoing_mask] / outgoing_count[outgoing_mask].to(
                device=edge_repr.device,
                dtype=edge_repr.dtype,
            ).unsqueeze(dim=-1)

        z = self._activation(h0 + self.theta_in(incoming_summary) + self.theta_out(outgoing_summary))
        recon_loss = self._edge_reconstruction_loss(z=z, adjacency=adjacency)
        return z, recon_loss

    def forward(
        self,
        x: torch.FloatTensor,
        view_adjs: dict[str, torch.Tensor],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Encode all relation-sequence views and return representations plus losses."""
        zs = []
        recon_losses = []
        for adjacency in view_adjs.values():
            adjacency = adjacency.to(device=x.device)
            z_phi, recon_loss = self._encode_one(adjacency=adjacency, x=x)
            zs.append(z_phi)
            recon_losses.append(recon_loss)

        z = torch.stack(zs, dim=0).mean(dim=0)
        recon_loss = torch.stack(recon_losses).mean() if recon_losses else z.new_zeros(())
        center = z.mean(dim=0, keepdim=True)
        scatter = (z - center).pow(2).sum(dim=-1).mean()
        # The scatter term is used as an anti-collapse pressure, hence the
        # negative sign. Keep it bounded, otherwise models whose scores depend
        # strongly on entity norms (notably RotatE) can drive this auxiliary
        # loss to enormous negative values and destabilize the KGC objective.
        scatter_loss = -scatter / (1.0 + scatter.detach())
        return z, recon_loss, scatter_loss


# Backward-compatible alias for older imports.
SharedMetaPathEncoder = SharedNonBacktrackingSpectralEncoder
