## Module Implementation

Structure Highlights:

- 1x1 preprocessing: `cv1` maps input channels to 2c and splits into two branches via `chunk`.
- Main branch: `RepConv(3x3)` + several `Conv(3x3)` for progressive refinement, then a final `1x1` for consolidation.
- Concatenation and fusion: concatenate the preserved branch with all stage outputs from the main branch, then fuse with a `1x1` convolution.
- Key hyperparameters: hidden ratio `e` and middle ratio `scale` decouple capacity and cost; inference-time re-parameterization is supported (`model_fuse_test`).

Notation: hidden channels $c=\lfloor e\cdot c_2 \rfloor$, middle channels $m=\lfloor \text{scale}\cdot c \rfloor$. Multiple 3x3 layers are applied only on $m$, significantly reducing compute.

## Core Ideas

1) RepConv x Ghost-style lightweight backbone

- Centered on 1x1 splitting + `RepConv(3x3)`, leveraging RepConv's structural redundancy and inference-time fusion for a "strong in training, simple at inference" design; compatible with Ghost-style efficiency by generating redundant features in narrow channels to reduce true 3x3 cost.

2) Dynamic channel split and stage-wise refinement

- `cv1` expands to 2c and splits: one branch preserves robust low-level features, the other runs a multi-stage 3x3 refinement chain. Stage outputs are concatenated to form a hierarchical semantic pyramid that preserves texture while strengthening semantics.

3) Low-cost multi-stage aggregation

- Instead of stacking 3x3 layers in high-dimensional space, computations are performed on the reduced middle channels $m$, followed by a final 1x1 fusion. This decouples "expressiveness" from "compute cost" and enables fine-grained tuning via `scale`/`e`.

4) Training vs. inference dual-mode optimization

- Training keeps redundant branches and deeper nonlinearity; inference fuses to an equivalent shallow structure for lower latency and easier deployment.

5) Compatibility with bidirectional pyramids

- Naturally fits bidirectional paths like FDPN/PAN: repeatedly used as a "lightweight refinement + aggregation" unit at P3/8 and P5/32 to improve cross-level information flow and gradient stability, especially for small or low-contrast targets.
