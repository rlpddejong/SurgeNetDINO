# 🧠 SurgeNetDINO: Towards Effective Surgical Representation Learning with DINO Models

This repository contains the official code and pretrained model weights for the paper:

**_“Towards Effective Surgical Representation Learning with DINO Models”_**  
**(Under Review for MIDL 2026)**

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

## 🙏 Acknowledgements

We would like to thank the authors and maintainers of the original DINO repositories for making their work publicly available:

- [DINOv1](https://github.com/facebookresearch/dino)  
- [DINOv2](https://github.com/facebookresearch/dinov2)  
- [DINOv3](https://github.com/facebookresearch/dinov3)  

Their open-source contributions provided the foundation for our work on surgical representation learning.

