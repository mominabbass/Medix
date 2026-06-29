# Medix: Out-of-Distribution Detection from Unlabeled Wild Data via Robust Gradient Statistics

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2510.06505)

## Installation
To setup the anaconda environment, simply run the following command:
```
conda env create -f setup_environment.yaml
```

After installation is complete, run:
```
conda activate data_eval
```

## Setup
### Dataset Preparation

**CIFAR-10 / CIFAR-100**

- The dataloader will automatically download the dataset the first time you run the program.

**OOD Datasets**

- The OOD datasets used with CIFAR-100 as the in-distribution dataset include five OOD datasets: SVHN, PLACES365, LSUN-C, LSUN-R, and TEXTURES.
- For more details, please refer to Part 1 and 2 of the codebase [here](https://github.com/deeplearning-wisc/knn-ood).


### Download checkpoints and other files:
Download the model checkpoints and other files by following these steps:
1) Go to Google Drive link: [https://drive.google.com/drive/folders/1deoDLzaBMcfra2xzN8YBGU7naSLuiiY_?usp=sharing](https://drive.google.com/drive/folders/1deoDLzaBMcfra2xzN8YBGU7naSLuiiY_?usp=sharing).
2) Download the `saved_checkpoint` folder and place it in the current directory. This folder contains the necessary checkpoints for running the main experiments.
3) Download the `saved_data` folder and place it in the current directory. This folder contains the filtered data from the first stage needed for the second stage of OOD detector training.


## Stage 1: Outlier filtering from wild data:
To filter outliers from wild data for CIFAR100-SVHN as the InD-OOD pair, run the following command:
```
CUDA_VISIBLE_DEVICES=0 python ood_filtering/OOD_cifar100_svhn.py"
```
This will save the outlier data in `saved_data` folder. To run another InD-OOD pair, run the corresponding file from the `ood_filtering` folder. To use a different GPU on your machine, change the `0` in `CUDA_VISIBLE_DEVICES=0` to the GPU number, e.g. `CUDA_VISIBLE_DEVICES=1` if you want to use GPU 1.


## Stage 2: OOD detector training:
To run Stage-2 directly, you can bypass Stage-1 by downloading the outlier data as described above.

To train the OOD detector for CIFAR100-SVHN as the InD-OOD pair, first specify the `outlier_data_dir` and `outlier_data_filename` variables in the `train_detector/main.py` file to the location where your outlier data is saved. Then, specify the path to the downloaded checkpoint in the `train_detector/main.py` file. Then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python train_detector/main.py --dataset cifar100 --aux_out_dataset svhn --test_out_dataset svhn --num_class 100
```
`--dataset` parameter refers to the in-distribution training data.

`--aux_out_dataset` parameter specifies the OOD data in the unlabeled wild data. To use a different InD-OOD pair, feel free to modify the parameter names as needed.


## Citation

If you find our work or this repository useful, please consider giving it a star ⭐ and citing our paper.

```bibtex
@article{abbas2026medix,
  title={Medix: Robust Gradient Statistics for Out-of-Distribution Detection from Unlabeled Wild Data},
  author={Momin Abbas and Ali Falahati and Hossein Goli and Mohammad Mohammadi Amiri},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=jFjA24PBJx}
}
```

## Contact

Should you have any inquiries, feel free to reach out via email at momin.abbas1@ibm.com .
