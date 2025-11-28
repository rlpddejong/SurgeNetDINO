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

## 📚 Citation
If you use these models or the dataset in your work, please cite our paper:
```bibtex
tbd
```

## 📬 Contact

For questions or issues regarding this repository, please contact the corresponding author:

**Ronald L.P.D. de Jong**  
Email: r.l.p.d.d.jong@tue.nl  

## 🙏 Acknowledgements

We would like to thank the authors and maintainers of the original DINO repositories for making their work publicly available:

- [DINO v1](https://github.com/facebookresearch/dino)  
- [DINO v2](https://github.com/facebookresearch/dinov2)  
- [DINO v3](https://github.com/facebookresearch/dinov3)  

Their open-source contributions provided the foundation for our work on surgical representation learning.

