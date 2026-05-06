# Multi-Receptive Refined Supervision Framework for Robust Underwater Object Detection

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20046486-blue.svg)](https://doi.org/10.5281/zenodo.20046486)

**Important Notice:** 
This repository contains the official PyTorch implementation and dataset guidelines for the manuscript **"Multi-Receptive Refined Supervision Framework for Robust Underwater Object Detection"**. This manuscript is currently under review for publication in **The Visual Computer**. If you find our widely open-source code, architectural designs, or experimental results helpful for your academic research or engineering projects, we strongly encourage and kindly request you to cite our manuscript.

## Introduction
Object detection in complex underwater environments presents significant challenges due to severe visual degradation, including turbidity, low-contrast, and the prevalence of small covered targets. MRS-DEIM is a robust one-stage detection framework specifically engineered for underwater perception. It achieves an optimal tradeoff between detection accuracy and computational efficiency, achieving an mAP of 82.6 percent on the URPC2020 benchmark.

## Implementations of Key Algorithms
To facilitate readers in replicating our experiments and evaluating the results, our framework introduces three core architectural improvements. You can find the specific algorithm implementations in the corresponding directories:
*   **MRF-PAN (Multi-Receptive Field Pyramid Aggregation Network):** Located in `engine/mrfpan.py`. It employs a dual-stage multi-kernel interaction mechanism to adaptively fuse global semantic context with local fine-grained details.
*   **RG-ELAN (Rep Ghost Efficient Layer Aggregation Network):** Located in `engine/rgelan.py`. This encoder synergizes structural re-parameterization with optimized gradient paths to enhance feature discriminability in low-contrast scenarios.
*   **SO-DAL (Small Object-aware Dynamic Adaptive Loss):** Located in `engine/deim_criterion.py`. A custom loss function designed to dynamically prioritize high-value and hard-to-detect small targets during the optimization phase.

## Dependencies and Requirements
To effortlessly replicate our experiments, please ensure your system meets the following hardware and software requirements:
Please install the required packages using the provided text file:

```bash
pip install -r requirements.txt
```
## Dependencies and Requirements
To effortlessly replicate our experiments, please ensure your system meets the following hardware and software requirements. Please install the required packages using the provided text file:

```bash
pip install -r requirements.txt
```

## Dataset Guidelines
Our framework is trained and evaluated on two widely recognized public benchmarks for underwater object detection:
*   **URPC2020:** The standard dataset from the Underwater Robot Professional Contest 2020.
*   **DUO:** The large-scale Detection in Underwater Objects benchmark.

Please download these datasets from their official open-source repositories and organize them according to standard YOLO/COCO format in your dataset directory before initiating the training or evaluation scripts.

## Citation
If you use this code or find our framework helpful in your research, please kindly cite our paper currently under review at **The Visual Computer**:

```bibtex
@article{mrs_deim_tvc,
  title={Multi-Receptive Refined Supervision Framework for Robust Underwater Object Detection},
  journal={The Visual Computer},
}
```
