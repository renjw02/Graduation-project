# SESD

This folder contains code for training and evaluating SESD.



## Setup instructions

Create a virtual anaconda environment:
```sh
conda create -n your_env_name python=3.8.5
```
Active it and install the cuda version Pytorch:
```sh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install other required modules and tools:
```sh
pip install -r requirements.txt
python nltk_downloader.py
```

Create several folders:

```sh
mkdir eval_results
mkdir models
mkdir predictions
mkdir data
```

Copy NL2PQL data from `../datasets` to `./data`.

 

## Script

`training` and `evaluating` code are in folder `./script`

