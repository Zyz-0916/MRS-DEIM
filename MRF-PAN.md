We propose MRF-PAN as the detection neck. The core idea follows a two-stage "focus-then-diffuse" pyramid, combining small-kernel multi-scale depthwise convolutions, a lightweight Transformer encoder, and cross-level alignment. This strengthens small targets and edge details while introducing necessary global context, without a significant compute increase.

Overall pipeline (backbone 0 MRF-PAN 0 decoder):
- Unified channel alignment: project P3/P4/P5 to the same `hidden_dim` (e.g., 256) to remove channel/statistical mismatch.
- Lightweight Transformer encoder: apply one or a few global modeling layers on high-level features (e.g., P5) with 2D sinusoidal positional encodings (precomputable at inference) to capture long-range dependencies at very low cost.
- Two-stage focus-diffuse pyramid: use P4 as the "focus center" and diffuse both upward and downward; repeat twice to progressively stabilize small-target saliency and cross-level consistency.

## Core

1) FocusFeature: cross-scale alignment with "small-kernel multi-kernel" feature focusing
- Three-branch alignment: upsample P5, identity/1x1 for P4, and adaptive downsample for P3 (ADown with avg-pooling + branch conv/pooling), then fuse after spatial alignment.
- Multi-kernel depthwise integration: apply parallel depthwise convolutions with multiple kernel sizes (e.g., k in {3,5,7,9}) on the concatenated feature, then fuse with a pointwise convolution. This balances edge detail and context with minimal parameter/FLOP overhead.
- Residual aggregation: add multi-kernel features back to the original concatenation to improve robustness and reduce overfitting.

2) Two-stage diffusion pyramid: bidirectional diffusion from the focus center, with iterative refinement
- Stage 1: take P4 as the center, diffuse down to P5 (concat original P5, then refine with C2f), then diffuse up to P3 (concat original P3, then refine with C2f).
- Stage 2: re-apply FocusFeature on the Stage-1 P5/P4/P3, then repeat bidirectional diffusion and refinement (C2f). This "focus 0 diffuse 0 return 0 re-diffuse" loop mitigates instability, information loss, and small-target suppression from single-pass fusion.
- Lightweight efficiency: core compute is concentrated in depthwise conv and RG-ELAN (efficient bottleneck structure), balancing expressiveness with complexity.
