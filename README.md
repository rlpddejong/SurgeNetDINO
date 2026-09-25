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
Dice score for semantic segmentation, accuracy (%) for surgical phase recognition, and online inference efficiency on a single NVIDIA H100 GPU (parameters of the segmentation models). Best results per DINO version are shown in bold.

<table>
  <tr><th rowspan="2">DINO</th><th rowspan="2">Model</th><th rowspan="2">Pretraining</th><th colspan="2">Segmentation (Dice ↑)</th><th colspan="2">Phase recognition (Acc ↑)</th><th colspan="3">Efficiency</th></tr>
  <tr><th>CholecSeg8k</th><th>RAMIE-seg</th><th>AutoLaparo</th><th>RAMIE-phase</th><th>Params (M)</th><th>Seg. FPS</th><th>Phase FPS</th></tr>
  <tr><td rowspan="4">v1</td><td rowspan="2">ViT-S</td><td>ImageNet</td><td>0.66</td><td>0.57</td><td>81.8</td><td>74.1</td><td rowspan="2">23</td><td rowspan="2">222</td><td rowspan="2">104</td></tr>
  <tr><td>SurgeNetXL</td><td><b>0.73</b></td><td>0.62</td><td><b>85.3</b></td><td><b>77.4</b></td></tr>
  <tr><td rowspan="2">ViT-B</td><td>ImageNet</td><td>0.68</td><td>0.61</td><td>81.1</td><td>75.4</td><td rowspan="2">92</td><td rowspan="2">213</td><td rowspan="2">89</td></tr>
  <tr><td>SurgeNetXL</td><td><b>0.73</b></td><td><b>0.67</b></td><td>85.0</td><td>77.2</td></tr>
  <tr><td rowspan="6">v2</td><td rowspan="2">ViT-S</td><td>LVD-142M</td><td>0.70</td><td>0.60</td><td>83.0</td><td>76.6</td><td rowspan="2">23</td><td rowspan="2">227</td><td rowspan="2">92</td></tr>
  <tr><td>SurgeNetXL</td><td>0.66</td><td>0.59</td><td>84.5</td><td>76.0</td></tr>
  <tr><td rowspan="2">ViT-B</td><td>LVD-142M</td><td>0.71</td><td>0.67</td><td>85.3</td><td>77.4</td><td rowspan="2">90</td><td rowspan="2">224</td><td rowspan="2">75</td></tr>
  <tr><td>SurgeNetXL</td><td>0.75</td><td>0.73</td><td>85.9</td><td>78.2</td></tr>
  <tr><td rowspan="2">ViT-L</td><td>LVD-142M</td><td>0.63</td><td>0.70</td><td><b>86.1</b></td><td>77.9</td><td rowspan="2">311</td><td rowspan="2">123</td><td rowspan="2">35</td></tr>
  <tr><td>SurgeNetXL</td><td><b>0.77</b></td><td><b>0.79</b></td><td>84.1</td><td><b>80.8</b></td></tr>
  <tr><td rowspan="6">v3</td><td rowspan="2">ViT-S</td><td>LVD-1689M</td><td>0.73</td><td>0.63</td><td>81.1</td><td>75.4</td><td rowspan="2">23</td><td rowspan="2">153</td><td rowspan="2">69</td></tr>
  <tr><td>SurgeNetXL</td><td>0.74</td><td>0.69</td><td>82.2</td><td>74.4</td></tr>
  <tr><td rowspan="2">ViT-B</td><td>LVD-1689M</td><td>0.71</td><td>0.67</td><td>83.4</td><td>75.8</td><td rowspan="2">92</td><td rowspan="2">147</td><td rowspan="2">60</td></tr>
  <tr><td>SurgeNetXL</td><td>0.75</td><td>0.73</td><td>83.3</td><td>76.3</td></tr>
  <tr><td rowspan="2">ViT-L</td><td>LVD-1689M</td><td>0.76</td><td>0.70</td><td>86.0</td><td>77.8</td><td rowspan="2">314</td><td rowspan="2">93</td><td rowspan="2">31</td></tr>
  <tr><td>SurgeNetXL</td><td><b>0.78</b></td><td><b>0.74</b></td><td><b>86.4</b></td><td><b>78.0</b></td></tr>
</table>

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

