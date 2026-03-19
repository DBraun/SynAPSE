"""Algorithm-specific graph construction for the DX7 GNN.

The per-algorithm topology (which operator modulates which, feedback edge,
carriers) is read directly from the ``dexed`` package's ``algorithms`` table, so
this stays the single source of truth for DX7 FM structure.
"""

import jax
import numpy as np
from dexed import algorithms
from einops import rearrange
from jax import numpy as jnp
from jax.typing import ArrayLike


def _algorithm_graph_arrays() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-algorithm graph arrays from each algorithm's mod_matrix + feedback edge.

    Returns (edges [32, 2, 6], edge_types [32, 6], masks [32, 6], carrier_masks [32, 6]).
    Max 6 edges/graph: up to 5 modulation edges + 1 feedback edge (e.g. 0-indexed algos
    15, 16, 17). Shorter graphs are zero-padded and masked.
    """
    max_edges_per_graph = 6

    all_algo_edges = []
    all_algo_edge_types = []  # 0: modulation, 1: feedback
    all_algo_masks = []

    for algo_idx in range(32):
        algo = algorithms[algo_idx]
        mod_matrix = algo.mod_matrix  # (6, 6) int8: [i,j]=1 means op j modulates op i

        edges = []
        edge_types = []

        # Add modulation connections (type 0) from mod_matrix
        for i in range(6):
            for j in range(6):
                if mod_matrix[i, j] == 1:
                    src = j
                    dst = i
                    edges.append([src, dst])
                    edge_types.append(0)  # modulation

        # Add feedback edge (type 1). feedback_edge is (source, target);
        # most algorithms have self-loops, but algos 4 and 6 have cross-operator
        # feedback (op 3→5 and op 4→5 respectively).
        fb_src, fb_tgt = algo.feedback_edge
        edges.append([fb_src, fb_tgt])
        edge_types.append(1)  # feedback

        # Pad to max_edges_per_graph
        num_edges = len(edges)
        if num_edges > max_edges_per_graph:
            edges = edges[:max_edges_per_graph]
            num_edges = max_edges_per_graph

        edge_array = (
            np.array(edges, dtype=np.int32).T
            if edges
            else np.zeros((2, 0), dtype=np.int32)
        )

        edge_type_array = (
            np.array(edge_types, dtype=np.int32)
            if edge_types
            else np.zeros(0, dtype=np.int32)
        )

        if num_edges < max_edges_per_graph:
            padding = np.zeros((2, max_edges_per_graph - num_edges), dtype=np.int32)
            edge_array = np.concatenate([edge_array, padding], axis=1)
            type_padding = np.zeros(max_edges_per_graph - num_edges, dtype=np.int32)
            edge_type_array = np.concatenate([edge_type_array, type_padding])

        mask = np.concatenate(
            [
                np.ones(num_edges, dtype=bool),
                np.zeros(max_edges_per_graph - num_edges, dtype=bool),
            ]
        )

        all_algo_edges.append(edge_array)
        all_algo_edge_types.append(edge_type_array)
        all_algo_masks.append(mask)

    # Carrier masks per algorithm (boolean, identifies carrier operators)
    all_carrier_masks = []
    for algo_idx in range(32):
        carrier_mask = np.zeros(6, dtype=bool)
        for carrier_op in algorithms[algo_idx].carriers:
            carrier_mask[carrier_op] = True
        all_carrier_masks.append(carrier_mask)

    return (
        jnp.array(all_algo_edges),  # [32, 2, 6]
        jnp.array(all_algo_edge_types),  # [32, 6]
        jnp.array(all_algo_masks),  # [32, 6]
        jnp.array(all_carrier_masks),  # [32, 6]
    )


def _fully_connected_graph_arrays() -> (
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Fully-connected 6-node graph, identical for every algorithm (algorithm ignored).

    30 directed modulation edges — every ordered pair ``(src=j, dst=i)`` with ``i != j``,
    all weight 1.0 (type 0). No self-loops and no feedback edges, so feedback is absent from
    the graph structure. Every operator is a carrier (output sums all 6). The per-algorithm
    axis is kept (broadcast) so the shared selection/vmap path below is unchanged.
    """
    max_edges_per_graph = 30

    fc_edges = np.array(
        [[j, i] for i in range(6) for j in range(6) if j != i], dtype=np.int32
    ).T  # [2, 30]  (src=j -> dst=i)

    all_algo_edges = jnp.broadcast_to(
        jnp.asarray(fc_edges), (32, 2, max_edges_per_graph)
    )
    all_algo_edge_types = jnp.zeros(
        (32, max_edges_per_graph), dtype=jnp.int32
    )  # all mod
    all_algo_masks = jnp.ones((32, max_edges_per_graph), dtype=bool)
    all_carrier_masks = jnp.ones((32, 6), dtype=bool)  # every operator is a carrier

    return all_algo_edges, all_algo_edge_types, all_algo_masks, all_carrier_masks


def build_graph_edges(
    algorithm_indices: ArrayLike,
    feedback_params: ArrayLike,
    graph_mode: str = "algorithm",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build edge indices and weights for the batch.

    Args:
        algorithm_indices: Algorithm index for each item [batch_size,].
        feedback_params: Feedback intensity parameter [batch_size,] (pre-scaled by the
            encoder). Unused in ``fully_connected`` mode, which has no feedback edges.
        graph_mode: ``"algorithm"`` (default) uses each algorithm's mod_matrix + feedback
            edge; ``"fully_connected"`` ignores the algorithm and uses a fixed 30-edge
            fully-connected graph (ablation).

    Returns:
        Tuple of (edge_index, edge_weights, edge_mask, carrier_mask) where:
        - edge_index: [2, batch_size * max_edges]
        - edge_weights: [batch_size * max_edges] (1.0 for modulation, scaled feedback for feedback)
        - edge_mask: [batch_size * max_edges] boolean mask
        - carrier_mask: [batch_size, 6] carrier mask
    """
    batch_size = algorithm_indices.shape[0]

    if graph_mode == "algorithm":
        all_algo_edges, all_algo_edge_types, all_algo_masks, all_carrier_masks = (
            _algorithm_graph_arrays()
        )
    elif graph_mode == "fully_connected":
        all_algo_edges, all_algo_edge_types, all_algo_masks, all_carrier_masks = (
            _fully_connected_graph_arrays()
        )
    else:
        raise RuntimeError(
            f"Unknown graph_mode: {graph_mode!r}. Expected 'algorithm' or 'fully_connected'."
        )

    # Use vmap to select edges and create weights for each batch item.
    @jax.vmap
    def select_edges(algo_idx, batch_idx, feedback_param):
        edges = all_algo_edges[algo_idx]  # [2, max_edges]
        edge_types = all_algo_edge_types[algo_idx]  # [max_edges]
        mask = all_algo_masks[algo_idx]  # [max_edges]
        carrier_mask = all_carrier_masks[algo_idx]  # [6] boolean

        # Add node offset (6 nodes per batch item)
        node_offset = batch_idx * 6
        edges = edges + node_offset

        # Modulation edges (type 0): weight 1.0. Feedback edges (type 1): weight =
        # feedback_param (already remapped and scaled by the encoder). Fully-connected
        # mode has no type-1 edges, so every weight is 1.0.
        edge_weights = jnp.where(
            edge_types == 1,
            feedback_param,
            1.0,
        )

        return edges, edge_weights, mask, carrier_mask

    batch_indices = jnp.arange(batch_size)
    edges_list, edge_weights_list, masks_list, carrier_mask = select_edges(
        algorithm_indices.astype(jnp.int32),
        batch_indices,
        feedback_params,
    )

    # Reshape to combine batch dimension
    edge_index = rearrange(
        edges_list, "b two e -> two (b e)", two=2
    )  # [2, B*max_edges]
    edge_weights = rearrange(edge_weights_list, "b e -> (b e)")  # [B*max_edges]
    edge_mask = rearrange(masks_list, "b e -> (b e)")  # [B*max_edges]

    return edge_index, edge_weights, edge_mask, carrier_mask
