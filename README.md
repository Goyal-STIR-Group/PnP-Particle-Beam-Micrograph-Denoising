# Particle Beam Micrograph Denoising with Plug-and-play Methods

[![DOI](https://zenodo.org/badge/641179077.svg)](https://zenodo.org/badge/latestdoi/641179077)

This repository contains the code used to produce results in the paper "Denoising Particle Beam Micrographs with Plug-and-Play Methods" by Minxu Peng, Ruangrawee Kitichotkul, Sheila W. Seidel, Christopher Yu, and Vivek K. Goyal.

## Abstract

In a particle beam microscope, a raster-scanned focused beam of particles interacts with a sample to generate a secondary electron (SE) signal pixel by pixel. Conventionally formed micrographs are noisy because of limitations on acquisition time and dose. Recent work has shown that estimation methods applicable to a time-resolved measurement paradigm can greatly reduce noise, but these methods apply pixel by pixel without exploiting image structure. The raw SE data can be modeled with a compound Poisson (Neyman Type A) likelihood, which implies data variance that is signal-dependent and greater than the variation in the underlying particle–sample interaction. These statistical properties make methods that assume additive white Gaussian noise ineffective. This paper introduces methods for particle beam micrograph denoising that use the plug-and-play framework to exploit image structure while being applicable to the unusual data likelihoods of this modality. Approximations of the data likelihood that vary in accuracy and computational complexity are combined with denoising by total variation regularization, BM3D, and DnCNN. Methods are provided for both conventional and time-resolved measurements. In simulations representative of helium ion microscopy and scanning electron microscopy, significant improvements in root mean-squared error (RMSE), structural similarity index measure (SSIM), and qualitative appearance are obtained. Average reductions in RMSE are by factors ranging from 2.24 to 4.11.

## Requirements

This code was tested with Python version 3.9.0. It requires the following packages: `numpy`, `matplotlib`, `pandas`, `sigpy`, `pytorch`, `scikit-image`, `Pillow`, `bm3d`, `jupyter`. Note that `setup.sh` also uses the `gdown` package to download from Google drive. You can install these packages manually, or install with `conda` using `environment.yml` as follow.

```bash
conda env create -f environment.yml
```

Note that `pytorch` listed in `environment.yml` has no support for GPU computing. If you wish to enable GPU computing, please manually install `pytorch` for an appropriate CUDA platform -- see https://pytorch.org/.

## Files

* `exp.py`: **(Important)** measurement simulation and denoising experiment code.
* `micrograph.py`: **(Important)** implementation of measurement simulation and denoising methods.

* `dncnn.py`: implementation of DnCNN.
* `util.py`: utility functions, such as error metrics.
* `demo.ipynb`: a demo for how to use `micrograph.py`.
* `lambda.ipynb`: an experiment about how optimal regularization strength changes with measurement parameters.
* `convergence.ipynb`: an experiment about convergence of the proposed PnP methods.
* `bash`
  * `demo.sh`: a demo bash file for running `exp.py`.
  * `setup.sh`: extract the dataset, download DnCNN weights, and make a result directory.
* `data`
  * `solvers`: contains an example of the solver specification `.csv` file as well as the solver specifications for HIM and SEM simulations as presented in our paper.
  * `spongeDataset.tar.gz`: compressed dataset file.

## Usage

We assume that the shell is `bash`. If you use a different shell, please modify the commands accordingly.

### Setup

First, please install the required Python packages, as explained in the Requirements section. Then, set up the directory by running

```
bash bash/setup.sh
```

This setup script will extract the dataset, make a directory for saving results, and download the DnCNN weights. If `gdown` fails to download the DnCNN weights, please download it manually with this [link](https://drive.google.com/file/d/1WkArJReKKGUX3TQIWzumnPch319C906m/view?usp=share_link), extract it, copy the model weights in the `checkpoints` directory to a `model` directory, and optionally rename the weights file. Please see `setup.sh` for details.

### Running Experiments

To run the particle beam microscope measurement and denoising simulations, please run `exp.py` using appropriate command line arguments as explained in the `main()` function of `exp.py`. You can also run the following command to see the explanations.

```
python exp.py --help
```

We provide an example of a command to run `exp.py` in `bash/demo.sh`. You can run the script as follow.

```
bash bash/demo.sh
```

We also provide the scripts to reproduce results in our paper -- `bash/himExp.sh` for HIM simulation and `bash/semExp.sh` for SEM simulation. 

Note that `exp.py` requires a `.csv` file to specify the solver details, as in the `--solverdir` argument. The first line must be the header: `method,f,g,rho,sigma`, and the following lines values of these fields. The fields should be as follow.

* `method`: can be admm, pgm, or fista.
* `f`: can be gaussian, oracle, qm, lqm, or tr.
* `g`: can be tv, bm3d, or dncnn.
* `rho`: 
  * For ADMM `rho` means the ADMM parameter $\rho$
  * For PGM or FISTA, `rho` means the step size for the gradient descent data-fidelity update $\gamma$.
* `sigma`: 
  * For TV denoiser, the `weight` in the argument of `denoise_tv_chambolle` is 0.5 * `sigma`.
  * For BM3D denoiser, `sigma` is the noise strength given to `bm3d.bm3d`, which is the `sigma_psd` argument.
  * For DnCNN denoiser, `sigma` is the denoiser scaling $\mu$.

The results saved are as follow:

* The results of an experiment are summarized in `results.csv` at the directory specified by the `savedir` argument.
* `imageName-data.npz` contains the measurement simulation results, including
  * `yTr`: time-resolved measurement.
  * `MConv`: (conventional) number of incident ions $M$.
  * `etaInit`: initialization of $\eta$, which is the conventional estimate $y / \lambda$.
  * `etaGt`: the ground truth $\eta$, which is simply the image with pixel values scale and shifted to the appropriate range, e.g. $[2, 8]$.
* `imageName-number.npz` contains the results for the image `imageName` using the solver setting in the index `number` in the `solvers.csv` file. For example, the first setting (the second line in `solver.csv` since the first line is reserved for the header) of image `plane` will be listed as `plane-0`. This file contains
  * `mse` and `snr` at each iteration.
  * final reconstructed $\eta$.
  * also all intermediate variables if `--fullLogging` is flagged.
* `imageName-number.png` the image `imageName` reconstructed using the method indexed `number` in `solvers.csv`.

### Code example

The implementation of our proposed denoising methods is in `micrograph.py`. The notebook `demo.ipynb` provides a simple example of how to use `micrograph.py`.

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2208.14256,
  doi = {10.48550/ARXIV.2208.14256},
  url = {https://arxiv.org/abs/2208.14256},
  author = {Peng, Minxu and Kitichotkul, Ruangrawee and Seidel, Sheila W. and Yu, Christopher and Goyal, Vivek K},
  keywords = {Medical Physics (physics.med-ph), FOS: Physical sciences, FOS: Physical sciences},
  title = {Denoising Particle Beam Micrographs with Plug-and-Play Methods},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Acknowledgements

The images in `spongeDataset.tar.gz` are crops from the "Porous_Sponge" NFFA-EUROPE SEM dataset [1].

Our implementation of the plug-and-play methods in `micrograph.py` heavily rely on abstractions provided by the `sigpy` package [(link)](https://sigpy.readthedocs.io/en/latest/).

## References

[1] R. Aversa, M. H. Modarres, S. Cozzini, and R. Ciancio, “NFFAEUROPE - SEM dataset,” 2018. [Online]. Available: https://b2share.eudat.eu/records/19cc2afd23e34b92b36a1dfd0113a89f

