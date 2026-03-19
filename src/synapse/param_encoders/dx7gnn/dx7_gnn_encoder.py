"""GNN-based encoder for DX7 synthesizer parameters.

This encoder models the DX7 synthesizer as a graph with 6 operator nodes,
handling all 32 algorithms through topology-aware message passing.

See dx7_features.py for the preset flat array layout and feature extraction.
"""

import argbind
import jax
from audiotree import AudioTree
from einops import rearrange
from flax import nnx
from flax.nnx import Rngs
from jax import numpy as jnp

from synapse.activations import parse_activation
from synapse.base_module import SaveLoadModule
from synapse.slap import MLP

from ..dx7_features import (
    NODE_INPUT_DIM_NO_OL,
    extract_node_features_no_ol,
    extract_output_levels,
)
from .build_graph_edges import build_graph_edges
from .fm_message_passing import FMMessagePassing
from .monotonic_remapping import MonotonicRemapping


@argbind.bind()
class DX7GNNEncoder(SaveLoadModule):
    """Graph Neural Network encoder for DX7 synthesizer parameters.

    This encoder processes DX7 parameters by modeling the synthesizer as a graph,
    where each operator is a node and connections represent modulation/feedback/carrier
    relationships that vary by algorithm.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        out_features: int = 512,
        num_layers: int = 8,
        dropout_rate: float = 0.3,
        activation: str = "gelu_tanh",
        use_weight_remapping: bool = True,
        remapping_num_knots: int = 40,
        share_gnn_weights: bool = True,
        input_ratios: list[float] = None,
        output_ratios: list[float] = None,
        use_scan: bool = False,
        value_mask_rate: float = 0.0,
        feedback_scale_mode: str = "auto",
        include_feedback_feature: bool = False,
        graph_mode: str = "algorithm",
        rngs: Rngs = None,
    ):
        """Initialize the DX7 GNN encoder.

        Args:
            hidden_dim: Hidden dimension for GNN layers
            out_features: Output embedding dimension
            num_layers: Number of GNN layers (temporal depth)
            dropout_rate: Dropout rate for Output MLP
            activation: Activation name for parse_activation (default "gelu_tanh")
            use_weight_remapping: Whether to use learned monotonic remapping for edge weights and OL
            remapping_num_knots: Number of knots for output level remapping (edge weight remapper is fixed at 10)
            share_gnn_weights: Share weights across GNN layers (1=shared, 0=per-layer)
            input_ratios: Multipliers for hidden_dim in input_module intermediate layers (default [1, 1])
                         e.g., [0.5, 2] creates bottleneck [node_input_dim, 0.5*hidden_dim, 2*hidden_dim, hidden_dim]
            output_ratios: Multipliers for hidden_dim in output_module intermediate layers (default [1, 1])
                          e.g., [0.5, 2] creates bottleneck [hidden_dim, 0.5*hidden_dim, 2*hidden_dim, output_dim]
            use_scan: Use nnx.scan for memory efficiency (only with share_gnn_weights=1)
            value_mask_rate: Probability of replacing each operator's features with a learned
                [MASK] embedding during training (0.0 = no masking). Inspired by PEACE's
                parameter masking which enables blind topology identification.
            feedback_scale_mode: How the feedback edge weight is scaled. "auto" (default)
                reproduces the original behavior (learned remapper * learnable scale when
                use_weight_remapping, else a fixed 0.5). "half" freezes it to 0.5 * raw
                feedback (the learned map's initialization); "zero" removes feedback from
                the graph. "half"/"zero" instantiate no feedback params.
            include_feedback_feature: If False, drop feedback from the global node features
                (48 -> 47 input dims). GNN-only "remove feedback global param" ablation.
            graph_mode: "algorithm" (default) uses each algorithm's topology; "fully_connected"
                ignores the algorithm and uses a fixed 30-edge fully-connected graph (ablation).
            rngs: Random number generators. Defaults to ``nnx.Rngs(0)`` so the model
                can be rebuilt with ``cls()`` under ``nnx.eval_shape`` during loading.
        """
        if rngs is None:
            rngs = Rngs(0)

        self.hidden_dim = hidden_dim
        self.output_dim = out_features
        self.num_layers = num_layers
        self.use_weight_remapping = use_weight_remapping
        self.share_gnn_weights = share_gnn_weights
        self.use_scan = use_scan
        self.value_mask_rate = value_mask_rate
        self.deterministic = False
        self.feedback_scale_mode = feedback_scale_mode
        self.include_feedback_feature = include_feedback_feature
        self.graph_mode = graph_mode

        # Set default ratios for backwards compatibility
        if input_ratios is None:
            input_ratios = [4]

        if isinstance(activation, str):
            activation = parse_activation(activation)
        if output_ratios is None:
            output_ratios = [4]

        # Input projection: raw node features -> hidden_dim (48 dims, or 47 when the feedback
        # global feature is removed). No dropout here: input is low-dimensional and reused as
        # original_features in every GNN layer.
        node_input_dim = NODE_INPUT_DIM_NO_OL - (0 if include_feedback_feature else 1)
        input_dims = (
            [node_input_dim]
            + [int(hidden_dim * ratio) for ratio in input_ratios]
            + [hidden_dim]
        )
        self.input_module = MLP(
            input_dims,
            activation=activation,
            normalization="layer",
            last_layer=None,
            dropout_rate=0.0,
            bias=None,
            out_bias=False,
            rngs=rngs,
        )

        # Create GNN layers (shared or per-layer)
        if self.share_gnn_weights:
            # Single layer with shared weights (backward compatible)
            self.gnn_layer = FMMessagePassing(
                in_features=hidden_dim,
                out_features=hidden_dim,
                rngs=rngs,
            )
            self.gnn_layers = None
        else:
            # Independent parameters per layer (more expressive)
            gnn_layers = []
            for i in range(num_layers):
                layer = FMMessagePassing(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    rngs=rngs,
                )
                gnn_layers.append(layer)
            self.gnn_layers = nnx.List(gnn_layers)
            self.gnn_layer = None

        # Output projection: carrier-sum hidden_dim -> embedding out_features
        output_dims = [hidden_dim] + [
            int(hidden_dim * ratio) for ratio in output_ratios
        ]
        if output_dims[-1] != out_features:
            output_dims.append(out_features)

        self.output_module = MLP(
            output_dims,
            activation=activation,
            normalization="layer",
            last_layer=None,
            dropout_rate=dropout_rate,
            bias=None,
            out_bias=False,
            rngs=rngs,
        )

        # Output-level remapping (independent of feedback handling).
        if self.use_weight_remapping:
            self.output_level_remapper = MonotonicRemapping(
                num_knots=remapping_num_knots
            )
        # Feedback edge-weight remapper + learnable scale. Only the "auto" mode learns these;
        # the "half"/"zero" ablations use a fixed multiplier and instantiate no feedback params.
        if self.use_weight_remapping and self.feedback_scale_mode == "auto":
            self.edge_weight_remapper = MonotonicRemapping(num_knots=10)
            # Learnable feedback scale: sigmoid(0) = 0.5 (DX7 default).
            # Temperature 0.1 makes this update ~10x slower than other params.
            self.feedback_scale_logit = nnx.Param(jnp.zeros(()))

        # Learned mask embedding for operator-level masking during training
        if self.value_mask_rate > 0:
            self.mask_embedding = nnx.Param(rngs.params.normal((hidden_dim,)) * 0.02)
            self.mask_rngs = nnx.Rngs(rngs.params())

    def get_learned_node_features(self, audio_tree: AudioTree) -> jax.Array:
        """Return shape [B, 6, D] containing learned features for each of the 6 operators in the graph."""
        flat_params = audio_tree.extras["params"]  # [B, 145]

        # Extract node features without output_level (48-dim, or 47 with feedback removed;
        # OL is used for gating instead).
        operator_features = extract_node_features_no_ol(
            flat_params, include_feedback=self.include_feedback_feature
        )  # [B, 6, 48 or 47]

        # Project to hidden dimension
        operator_features = self.input_module(operator_features)  # [B, 6, hidden_dim]

        return operator_features

    def __call__(self, audio_tree: AudioTree) -> jax.Array:
        """Forward pass through the GNN encoder.

        Args:
            audio_tree: AudioTree with extras containing:
                - params: [batch_size, 145] flat parameters from Preset.to_array()
                - algorithm: [batch_size,] algorithm indices (0-31)

        Returns:
            Embeddings of shape [batch_size, output_dim]
        """
        flat_params = audio_tree.extras["params"]
        algorithm_indices = audio_tree.extras["algorithm"]

        B = flat_params.shape[0]

        # Feedback is global continuous index 0 in the Preset flat array
        feedback_params = flat_params[:, 0]  # [B,] in range [0, 1]

        # Scale feedback params (the feedback edge weight).
        # - "auto": learned remapper * learnable scale (init 0.5) when use_weight_remapping,
        #   else the fixed DX7 default of 0.5 — reproduces the original behavior exactly.
        # - "half": fixed 0.5 * raw feedback (freeze to the learned map's initialization).
        # - "zero": remove feedback from the graph entirely.
        if self.feedback_scale_mode == "auto":
            if self.use_weight_remapping:
                feedback_params = self.edge_weight_remapper(feedback_params)
                feedback_scale = nnx.sigmoid(0.1 * self.feedback_scale_logit)
                feedback_params = feedback_params * feedback_scale
            else:
                feedback_params = feedback_params * 0.5
        elif self.feedback_scale_mode == "half":
            feedback_params = feedback_params * 0.5
        elif self.feedback_scale_mode == "zero":
            feedback_params = feedback_params * 0.0
        else:
            raise RuntimeError(
                f"Unknown feedback_scale_mode: {self.feedback_scale_mode!r}. "
                "Expected 'auto', 'half', or 'zero'."
            )

        # Build graph edges
        # - edge_index: [2, batch_size * max_edges]
        # - edge_weights: [batch_size * max_edges] (1.0 for modulation, scaled feedback for feedback)
        # - edge_mask: [batch_size * max_edges] boolean mask
        # - carrier_mask: [batch_size, 6] carrier mask
        edge_index, edge_weights, _, carrier_mask = build_graph_edges(
            algorithm_indices, feedback_params, graph_mode=self.graph_mode
        )

        # Extract output levels for OL gating
        output_levels = extract_output_levels(flat_params)  # [B, 6] in [0, 1]
        if self.use_weight_remapping:
            output_levels = self.output_level_remapper(output_levels)
        output_levels_flat = rearrange(output_levels, "B six -> (B six)", six=6)

        node_features_flat = self.get_learned_node_features(audio_tree)  # [b, 6, d]

        # Operator-level masking: replace random operators with learned [MASK] embedding
        if self.value_mask_rate > 0 and not self.deterministic:
            # True where we replace with mask embedding
            mask = self.mask_rngs.bernoulli(self.value_mask_rate, shape=(B, 6, 1))
            mask_emb = jnp.broadcast_to(
                self.mask_embedding[...][None, None, :], node_features_flat.shape
            )
            node_features_flat = jnp.where(mask, mask_emb, node_features_flat)

        # Reshape for GNN processing [B*6, hidden_dim]
        node_features_flat = rearrange(
            node_features_flat, "B six D -> (B six) D", B=B, six=6
        )

        # Apply GNN layers
        x = jnp.zeros_like(node_features_flat)

        if self.share_gnn_weights:
            # Use single layer repeatedly (current behavior)
            if self.use_scan:
                # Memory-efficient scan version
                @nnx.scan(length=self.num_layers, in_axes=(nnx.Carry,))
                def scan_loop(carry):
                    x_msg, gnn_layer = carry
                    x_msg = gnn_layer(
                        x_msg,
                        edge_index,
                        edge_weight=edge_weights,
                        original_features=node_features_flat,
                        output_level=output_levels_flat,
                    )
                    return (x_msg, gnn_layer), None

                (x, _), _ = scan_loop((x, self.gnn_layer))
            else:
                # Simple loop
                for _ in range(self.num_layers):
                    x = self.gnn_layer(
                        x,
                        edge_index,
                        edge_weight=edge_weights,
                        original_features=node_features_flat,
                        output_level=output_levels_flat,
                    )
        else:
            # Use different layer parameters per iteration
            for layer in self.gnn_layers:
                x = layer(
                    x,
                    edge_index,
                    edge_weight=edge_weights,
                    original_features=node_features_flat,
                    output_level=output_levels_flat,
                )

        # Reshape back to [B, 6, hidden_dim]

        node_outputs = rearrange(x, "(B six) D -> B six D", B=B, six=6)

        # Apply carrier weights to scale carrier operators by their Output Levels
        # carrier_mask: [B, 6] float weights (0.0 for non-carriers, 1.0 for carriers)
        carrier_features = node_outputs * carrier_mask[:, :, None]  # [B, 6, hidden_dim]

        carrier_sum = jnp.sum(carrier_features, axis=1)  # [B, hidden_dim]

        num_channels = 1
        if audio_tree.waveform is not None:
            num_channels = audio_tree.waveform.shape[1]
        elif "mel" in audio_tree.extras:
            # Mel format is (B, n_mels, n_frames, channels) — channel dim is last
            num_channels = audio_tree.extras["mel"].shape[-1]

        if num_channels == 1:
            # Single channel: return representation for carrier summation only
            x = carrier_sum
        else:
            # Multi-channel (6 operators + carrier sum = 7): return all representations
            carrier_sum = jnp.expand_dims(carrier_sum, axis=1)  # [B, 1, hidden_dim]
            x = jnp.concatenate(
                [node_outputs, carrier_sum], axis=1
            )  # [B, 7, hidden_dim]
            x = rearrange(x, "B seven out_dim -> (B seven) out_dim", B=B, seven=7)

        # Final projection to output dimension
        embeddings = self.output_module(x)  # [B*7, out_dim] or [B, out_dim]

        return embeddings


if __name__ == "__main__":
    # Test the encoder
    rngs = Rngs(42)

    # Create encoder
    encoder = DX7GNNEncoder(rngs=rngs)

    # Create fake AudioTree with 145-dim Preset params
    batch_size = 1
    audio_tree = AudioTree(
        jnp.zeros((batch_size, 1, 44100)),
        sample_rate=44100,
        extras={
            "params": jnp.full((batch_size, 145), 0.5),
            "algorithm": jnp.full((batch_size), 1).astype(jnp.int32),
        },
    )

    print(nnx.tabulate(encoder, audio_tree, depth=3, console_kwargs={"width": 200}))
