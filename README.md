# Reducing spatial aliasing in seismic imaging using deep learning superresolution.

The limited ability to capture high-resolution seismic data causes a spatial aliasing effect that disrupts later analyses that are performed on it.
We apply deep learning super-resolution in order to up-scale the data by a factor of two and remove the spatial aliasing.
Our modified VDSR model produces good qualitative and quantitative results on our test set and succeeds at removing the aliasing.

This repository contains code to train a superresolution network from scratch and apply it to new data.
Other scripts are included to perform the experiments and hyperparameter tuning as described in the accompanying article [link TBA].

## Dependencies

- [Python 3.7](https://pytorch.org/)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/)
- [matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)
- [Hyperopt](https://github.com/hyperopt/hyperopt)
- [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)

## Scripts

### Main

- train.py; trains a model on a given dataset.
- models.py; model class definitions for SRCNN, EDSR, and VDSR.
- dataset.py; implementation of the dataset class and transformations.

### Notebooks

__TBA__

### Tuning and experiments

- tuning.py; used for hyperparameter tuning.
- table_generation.py; compute the performance of models with different loss functions.
- less_data.py; trains a model on several subsets of the dataset to test the impact on performance.

### Miscellaneous

The remaining shell scripts are used for running the training and experiments on Surfsara's Lisa.

## Getting started

### Using the notebooks

__TBA__

### Without the notebooks

First, make sure you have high- and __corresponding__ low-resolution data prepared.
Data is expected to be stored in _.mat_ files.
For example, _data_20.mat_ and _data_10.mat_ for low- and high-resolution data, respectively.
Put these two files in a directory, for example _my_data/_.

Make sure you have the dependencies installed and run train.py, pointing to the right files;

```sh
$ python train.py --data_root my_data/ -x data_20 -y data_10
```

The default parameters should work reasonably well.
For more information on al the options, type 

```sh
$ python train.py -h
```
