# Multi-Receptive Refined Supervision Framework for Robust Underwater Object Detection


**Important Notice:** 
This repository contains the official PyTorch implementation and dataset guidelines for the manuscript **"Multi-Receptive Refined Supervision Framework for Robust Underwater Object Detection"**. This manuscript is currently under review for publication in **The Visual Computer**. If you find our widely open source code, architectural designs, or experimental results helpful for your academic research or engineering projects, we strongly encourage and kindly request you to cite our manuscript.

## Introduction
Object detection in complex underwater environments presents significant challenges due to severe visual degradation, including turbidity, low contrast, and the prevalence of small covered targets. MRS-DEIM is a robust one stage detection framework specifically engineered for underwater perception. It achieves an optimal tradeoff between detection accuracy and computational efficiency, achieving an mAP of 82.6 percent on the URPC2020 benchmark.

## Implementations of Key Algorithms
To facilitate readers in replicating our experiments and evaluating the results, our framework introduces three core architectural improvements. You can find the specific algorithm implementations in the corresponding directories:
*   **MRF-PAN (Multi Receptive Field Pyramid Aggregation Network):** Located in `engine/mrfpan.py`. It employs a dual stage multi kernel interaction mechanism to adaptively fuse global semantic context with local fine grained details.
*   **RG-ELAN (Rep Ghost Efficient Layer Aggregation Network):** Located in `engine/rgelan.py`. This encoder synergizes structural reparameterization with optimized gradient paths to enhance feature discriminability in low contrast scenarios.
*   **SO-DAL (Small Object aware Dynamic Adaptive Loss):** Located in `engine/deim_criterion.py`. A custom loss function designed to dynamically prioritize high value and hard to detect small targets during the optimization phase.

## Dependencies and Requirements
To effortlessly replicate our experiments, please ensure your system meets the following hardware and software requirements:
Please install the required packages using the provided text file:
```bash
pip install -r requirements.txt


@article{MRS_DEIM,
  title={Multi Receptive Refined Supervision Framework for Robust Underwater Object Detection},
  journal={The Visual Computer},
}
