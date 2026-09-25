# 🧠 SurgeNetDINO: Towards Effective Surgical Representation Learning with DINO Models

This repository contains the official code and pretrained model weights for the paper:

**_“Towards Effective Surgical Representation Learning with DINO Models”_**  
**(Accepted for Medical Imaging with Deep Learning (MIDL) 2026)**

**[Find the paper here!](https://openreview.net/pdf?id=6FoIDPKzRV)**

## ⚙️ Requirements
- Python 3.9+
- torch
- timm

## 🐍 Installation example using Conda
```bash
# Create a new conda environment with Python 3.9
conda create -n SurgeNetDINO python=3.9 -y

# Activate the environment
conda activate SurgeNetDINO

# Install PyTorch and timm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm
```

## 💾 Loading models
After installations, you can use `load_weights.py` to load the pretrained DINO models.

## ⬇️ Download the model weights here
Alternatively, you can download the model weights using the provided links below.

| Model   | Variant | Download |
|---------|---------|----------|
| **DINOv1** | ViT-s | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTs16_size224_SurgeNetXL.pth?download=true) |
| **DINOv1** | ViT-b | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTb16_size224_SurgeNetXL.pth?download=true) |
| **DINOv2** | ViT-s | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTs14_size336_SurgeNetXL.pth?download=true) |
| **DINOv2** | ViT-b | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTb14_size336_SurgeNetXL.pth?download=true) |
| **DINOv2** | ViT-l | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTl14_size336_SurgeNetXL.pth?download=true) |
| **DINOv3** | ViT-s | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTs16_size336_SurgeNetXL.pth?download=true) |
| **DINOv3** | ViT-b | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTb16_size336_SurgeNetXL.pth?download=true) |
| **DINOv3** | ViT-l | [Download](https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTl16_size336_SurgeNetXL.pth?download=true) |

## 📄 Pretraining configurations
The pretraining configurations can be found in ``dinov1_configs``, ``dinov2_configs``, and ``dinov3_configs``. Note that for ``dinov1``, we provide a `.sh` file instead of a configuration file following the original implementation. For further pretraining instructions, please refer to the original implementations of DINO: [DINOv1](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), and [DINOv3](https://github.com/facebookresearch/dinov3).

## 🎯 Finetuning
- **Semantic segmentation:** The code for finetuning the pretrained models on semantic segmentation can be found in [`segmentation_finetuning`](segmentation_finetuning), which is based on [EoMT](https://github.com/tue-mps/eomt). Additional Python packages are required, which can be installed with `pip install -r segmentation_finetuning/requirements.txt`.
- **Surgical phase recognition:** For finetuning on surgical phase recognition, please refer to [SurgPhaseBench](https://github.com/Yipinggggg/SurgPhaseBench).

## 📊 Main results
Vanilla ViT architectures trained from scratch are included as baselines. Best results per DINO version are shown in bold.

### Semantic segmentation
| DINO | Model | Pretraining | CholecSeg8k Dice ↑ | CholecSeg8k HD95 ↓ | RAMIE-seg Dice ↑ | RAMIE-seg HD95 ↓ |
|------|-------|-------------|--------------------|--------------------|------------------|------------------|
| – | ViT-S | No pretraining | 0.52 ± 0.05 | 100 ± 16 | 0.35 ± 0.04 | 133 ± 16 |
| – | ViT-B | No pretraining | 0.52 ± 0.07 | 101 ± 16 | 0.41 ± 0.06 | 125 ± 9 |
| – | ViT-L | No pretraining | 0.55 ± 0.10 | 102 ± 16 | 0.44 ± 0.05 | 123 ± 6 |
| v1 | ViT-S | ImageNet | 0.66 ± 0.09 | 65 ± 25 | 0.57 ± 0.04 | 73 ± 11 |
| v1 | ViT-S | SurgeNetXL | **0.73 ± 0.09** | **56 ± 19** | 0.62 ± 0.03 | 63 ± 6 |
| v1 | ViT-B | ImageNet | 0.68 ± 0.07 | 61 ± 23 | 0.61 ± 0.02 | 69 ± 9 |
| v1 | ViT-B | SurgeNetXL | **0.73 ± 0.10** | 57 ± 19 | **0.67 ± 0.03** | **58 ± 14** |
| v2 | ViT-S | LVD-142M | 0.70 ± 0.07 | 54 ± 19 | 0.60 ± 0.04 | 70 ± 7 |
| v2 | ViT-S | SurgeNetXL | 0.66 ± 0.06 | 55 ± 18 | 0.59 ± 0.03 | 67 ± 5 |
| v2 | ViT-B | LVD-142M | 0.71 ± 0.06 | 50 ± 18 | 0.67 ± 0.05 | 52 ± 6 |
| v2 | ViT-B | SurgeNetXL | 0.75 ± 0.10 | 44 ± 22 | 0.73 ± 0.04 | 46 ± 11 |
| v2 | ViT-L | LVD-142M | 0.63 ± 0.09 | 58 ± 13 | 0.70 ± 0.05 | 49 ± 6 |
| v2 | ViT-L | SurgeNetXL | **0.77 ± 0.09** | **38 ± 20** | **0.79 ± 0.04** | **40 ± 7** |
| v3 | ViT-S | LVD-1689M | 0.73 ± 0.06 | 52 ± 21 | 0.63 ± 0.06 | 59 ± 6 |
| v3 | ViT-S | SurgeNetXL | 0.74 ± 0.07 | 48 ± 17 | 0.69 ± 0.05 | 54 ± 8 |
| v3 | ViT-B | LVD-1689M | 0.71 ± 0.07 | 55 ± 24 | 0.67 ± 0.04 | 53 ± 10 |
| v3 | ViT-B | SurgeNetXL | 0.75 ± 0.09 | 42 ± 22 | 0.73 ± 0.05 | 55 ± 14 |
| v3 | ViT-L | LVD-1689M | 0.76 ± 0.09 | 34 ± 21 | 0.70 ± 0.04 | **46 ± 3** |
| v3 | ViT-L | SurgeNetXL | **0.78 ± 0.11** | **29 ± 15** | **0.74 ± 0.04** | **46 ± 12** |

### Surgical phase recognition
| DINO | Model | Pretraining | AutoLaparo Accuracy ↑ | AutoLaparo F1 ↑ | RAMIE-phase Accuracy ↑ | RAMIE-phase F1 ↑ |
|------|-------|-------------|-----------------------|-----------------|------------------------|------------------|
| – | ViT-S | No pretraining | 53.9 ± 14.8 | 45.8 ± 16.8 | 63.1 ± 14.4 | 52.1 ± 16.0 |
| – | ViT-B | No pretraining | 67.9 ± 18.3 | 59.5 ± 16.5 | 64.2 ± 12.6 | 51.0 ± 14.3 |
| – | ViT-L | No pretraining | 65.6 ± 15.5 | 58.1 ± 13.5 | 64.3 ± 14.5 | 52.3 ± 16.2 |
| v1 | ViT-S | ImageNet | 81.8 ± 9.4 | 70.4 ± 6.8 | 74.1 ± 9.3 | 65.3 ± 10.0 |
| v1 | ViT-S | SurgeNetXL | **85.3 ± 5.6** | **75.1 ± 5.2** | **77.4 ± 9.8** | **70.5 ± 11.8** |
| v1 | ViT-B | ImageNet | 81.1 ± 9.0 | 68.7 ± 8.4 | 75.4 ± 10.0 | 66.4 ± 10.9 |
| v1 | ViT-B | SurgeNetXL | 85.0 ± 9.1 | 70.6 ± 7.6 | 77.2 ± 10.6 | 70.1 ± 11.5 |
| v2 | ViT-S | LVD-142M | 83.0 ± 9.9 | 73.0 ± 11.0 | 76.6 ± 9.5 | 67.9 ± 9.9 |
| v2 | ViT-S | SurgeNetXL | 84.5 ± 8.6 | 69.6 ± 6.4 | 76.0 ± 11.0 | 66.3 ± 12.2 |
| v2 | ViT-B | LVD-142M | 85.3 ± 7.4 | 74.2 ± 8.2 | 77.4 ± 9.9 | 67.8 ± 10.4 |
| v2 | ViT-B | SurgeNetXL | 85.9 ± 8.3 | 74.0 ± 5.9 | 78.2 ± 10.2 | 71.1 ± 11.8 |
| v2 | ViT-L | LVD-142M | **86.1 ± 7.6** | **77.8 ± 8.9** | 77.9 ± 10.3 | 70.2 ± 11.4 |
| v2 | ViT-L | SurgeNetXL | 84.1 ± 8.4 | 70.5 ± 8.0 | **80.8 ± 9.1** | **73.7 ± 11.5** |
| v3 | ViT-S | LVD-1689M | 81.1 ± 10.0 | 69.7 ± 8.6 | 75.4 ± 10.8 | 63.3 ± 11.8 |
| v3 | ViT-S | SurgeNetXL | 82.2 ± 8.5 | 72.1 ± 8.0 | 74.4 ± 10.9 | 62.9 ± 11.7 |
| v3 | ViT-B | LVD-1689M | 83.4 ± 9.4 | 70.8 ± 10.1 | 75.8 ± 10.7 | 66.8 ± 10.5 |
| v3 | ViT-B | SurgeNetXL | 83.3 ± 8.5 | 73.0 ± 4.7 | 76.3 ± 10.5 | 66.7 ± 12.1 |
| v3 | ViT-L | LVD-1689M | 86.0 ± 8.6 | 75.0 ± 9.6 | 77.8 ± 9.2 | 69.8 ± 10.3 |
| v3 | ViT-L | SurgeNetXL | **86.4 ± 6.6** | **78.5 ± 6.7** | **78.0 ± 10.5** | **70.1 ± 11.8** |

### Inference efficiency
Online inference on a single NVIDIA H100 GPU. Params in millions (M), latency in milliseconds (ms).

| DINO | Model | Seg. Params (M) | Seg. Latency (ms) | Seg. FPS | Phase Params (M) | Phase Latency (ms) | Phase FPS |
|------|-------|-----------------|-------------------|----------|------------------|--------------------|-----------|
| v1 | ViT-S | 23 | 4.5 | 222 | 22 | 9.6 | 104 |
| v1 | ViT-B | 92 | 4.7 | 213 | 86 | 11.3 | 89 |
| v2 | ViT-S | 23 | 4.4 | 227 | 22 | 10.8 | 92 |
| v2 | ViT-B | 90 | 4.5 | 224 | 86 | 13.3 | 75 |
| v2 | ViT-L | 311 | 8.1 | 123 | 304 | 28.9 | 35 |
| v3 | ViT-S | 23 | 6.5 | 153 | 22 | 14.4 | 69 |
| v3 | ViT-B | 92 | 6.8 | 147 | 86 | 16.7 | 60 |
| v3 | ViT-L | 314 | 10.8 | 93 | 304 | 32.6 | 31 |

## 📚 Citation
If you use these models or the dataset in your work, please cite our paper:
```bibtex
@inproceedings{
  jong2026towards,
  title={Towards Effective Surgical Representation Learning with {DINO} Models},
  author={Ronald L.P.D. de Jong and Yiping Li and Tim J. M. Jaspers and Romy C. van Jaarsveld and Gino M. Kuiper and Franco Badaloni and Richard van Hillegersberg and Jelle P. Ruurda and Fons van der Sommen and Josien P.W. Pluim and Marcel Breeuwer},
  booktitle={Medical Imaging with Deep Learning},
  year={2026},
  url={https://openreview.net/forum?id=6FoIDPKzRV}
}
```

## 📬 Contact

For questions or issues regarding this repository, please contact the corresponding author:

**Ronald L.P.D. de Jong**  
Email: r.l.p.d.d.jong@tue.nl  

## ⚖️ License
- Code: MIT — see LICENSE (permissive; commercial use permitted).
- Pretrained model weights: CC-BY-NC-SA — non-commercial share-alike. The weights and any derivative models that include these weights are NOT cleared for commercial use. See LICENSE_MODELS for details and the precise license text.

## 🙏 Acknowledgements

We would like to thank the authors and maintainers of SurgeNetXL and the original DINO repositories for making their work publicly available:

- [SurgeNetXL](https://github.com/TimJaspers0801/SurgeNet)
- [DINOv1](https://github.com/facebookresearch/dino)  
- [DINOv2](https://github.com/facebookresearch/dinov2)  
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [EoMT](https://github.com/tue-mps/eomt)

Their open-source contributions provided the foundation for our work on surgical representation learning.

