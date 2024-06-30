[![arXiv](https://img.shields.io/badge/arXiv-2305.16404-b31b1b.svg)](https://arxiv.org/abs/2403.18201)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

## Few-shot Online Anomaly Detection and Segmentation
Official code of "Few-shot Online Anomaly Detection and Segmentation"

## Installation
Create a Conda environment with [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Since some pytorch API has been changed, please do **NOT** use the latest pytorch version!
```bash
conda create -n kng python=3.7
conda activate kng
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Running
You can train a model for different dataset by specifying `--data_path`: 

```bash
CUDA_VISIBLE_DEVICES=0 python Konline.py --data_path ../datasets/MVTec-AD
```

## Citation
If you find our work useful in your research, please consider citing:

``` shell script
@article{wei2024few,
  title={Few-shot online anomaly detection and segmentation},
  author={Wei, Shenxing and Wei, Xing and Ma, Zhiheng and Dong, Songlin and Zhang, Shaochen and Gong, Yihong},
  journal={Knowledge-Based Systems},
  pages={112168},
  year={2024},
  publisher={Elsevier}
}
```
