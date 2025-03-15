## System-level Prereq

If you don't have conda/miniconda, I recommend miniconda, which is a lightweight version of anaconda that has essential features.

You can get the installation [here](https://docs.anaconda.com/free/miniconda/).

Follow the instruction [here](https://stackoverflow.com/questions/76760906/installing-mamba-on-a-machine-with-conda) to set up Mamba, a fast environment solver for conda.

```
## prioritize 'conda-forge' channel
conda config --add channels conda-forge

## update existing packages to use 'conda-forge' channel
conda update -n base --all

## install 'mamba'
conda install -n base mamba
```

Note: technically, the mamba solver should behave the same as the default solver. However, there have been cases where dependencies
can not be properly set up with the default mamba solver. The following instructions have **only** been tested on mamba solver.

### Setup Conda

```bash
conda create -y -n human_policy python=3.11 && conda activate human_policy
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip install -r requirements.txt
cd human-policy
pip install -e .
pip install -e hdt/detr
```

### FAQ

1. If you encounter the following error:

```
ImportError: cannot import name 'packaging' from 'pkg_resources' 
```

It may be due to the version of setuptools. You can downgrade the version of setuptools by running the following command:

```bash
pip install setuptools==69.5.1
```
