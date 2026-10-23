# Original Code Repositories and Research Acknowledgements

> This repository accompanies an unpublished research implementation. The paper has not yet been publicly released. This file lists relevant prior methods and public reference repositories; a method-level link does **not** establish that the present source file was directly copied from that repository.

## Reference implementations

| Method | Related files | Reference code | Original research | Implementation note |
|---|---|---|---|---|
| EfficientNet | `model_eff2.py` | [tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019) | Backbone design; code provenance not confirmed |
| WTConv | `model_eff2.py` | [BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv) | Wavelet Convolutions for Large Receptive Fields (ECCV 2024) | WTConv2d and wavelet filters; check reuse licensing |
| SHViT / SHSA | `SHSA.py` | [ysj9909/SHViT](https://github.com/ysj9909/SHViT) | SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design (CVPR 2024) | Single-head self-attention module |
| CBAM | `Channel.py; Spatial.py` | [Jongchan/attention-module](https://github.com/Jongchan/attention-module) | CBAM: Convolutional Block Attention Module (ECCV 2018) | Channel/spatial attention inspired by CBAM |
| SENet | `model_eff2.py; SEAttention2.py` | [hujie-frank/SENet](https://github.com/hujie-frank/SENet) | Squeeze-and-Excitation Networks (CVPR 2018) | SqueezeExcitation and modified pooling-based SE-like block |
| SimAM | `model_eff2.py` | [ZjjConan/SimAM](https://github.com/ZjjConan/SimAM) | SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021) | Modified implementation: spatial Softmax rather than original Sigmoid |
| ECA-Net | `model_eff2.py` | [BangguWu/ECANet](https://github.com/BangguWu/ECANet) | ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2020) | Defined but not called in current forward path |
| SCNet / SCConv | `model_eff2.py` | [MCG-NKU/SCNet](https://github.com/MCG-NKU/SCNet) | Improving Convolutional Networks with Self-Calibrated Convolutions (CVPR 2020) | Defined but not called in current forward path |

## Implementation and attribution notes

- **EfficientNet:** The network uses EfficientNet-style MBConv blocks and width/depth scaling, with a custom three-input feature-extraction/fusion design.
- **WTConv:** `WTConv2d` and the wavelet transform routines align with the public WTConv method. Cite both the paper and the reference repository when publishing code.
- **SHSA:** `SHSA.py` implements single-head self-attention associated with SHViT; do not cite only the general Transformer literature.
- **CBAM-related modules:** `Channel.py` and `Spatial.py` use pooling-based channel/spatial attention related to CBAM; whether the exact source was copied cannot be established from these files alone.
- **Modified SE:** `SEAttention2.py` combines average, maximum and variance pooling, and a context convolution. It should not be described as an unmodified SENet block.
- **Modified SimAM:** `model_eff2.py` applies a spatial `Softmax` rather than the original SimAM's `Sigmoid`; label this as a modified or SimAM-inspired implementation.
- **Unused classes:** ECA and SCConv classes are defined in `model_eff2.py` but are not used in the shown model forward pass; they should not be claimed as active parts of the architecture.

## Unpublished manuscript statement

This repository contains research code associated with an unpublished manuscript. Further citation information will be provided if and when the manuscript becomes publicly available. No unpublished performance claims or invented bibliographic identifiers are provided here.

## Licensing reminder

Academic citation alone does not grant permission to redistribute third-party source code. Before publishing, verify the source provenance of any copied/adapted implementation, review each upstream LICENSE, preserve applicable notices, and choose a compatible license for your own contributions.

## Suggested GitHub acknowledgement

We acknowledge the authors of EfficientNet, WTConv, SHViT, CBAM, SENet, SimAM, ECA-Net, and SCNet for their contributions to the research community. Public reference implementations are linked above. Some modules in this repository are modified, and the linked repositories do not necessarily represent the exact origin of every code fragment.
