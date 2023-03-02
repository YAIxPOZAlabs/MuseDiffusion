<img src="https://i.ibb.co/z66nz7q/1.png">
<br>
<h2> How to run</h2>

<h3>0. Clone repository and cd</h3>

```bash
git clone https://github.com/YAIxPOZAlabs/MuseDiffusion.git
cd MuseDiffusion
```

<h3>1. Prepare environment and data</h3>

<h4>(Optional) Install python 3.8 <i>for Virtualenv usage</i></h4>

```bash
sudo apt update && \
sudo apt install -y software-properties-common && \
sudo add-apt-repository -y ppa:deadsnakes/ppa && \
sudo apt install -y python3.8 python3.8-distutils
```

<h4>With Virtualenv</h4>

```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv --python=python3.8
source venv/bin/activate
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

<h4>With Anaconda</h4>

```bash
conda env create -n python=3.8 MuseDiffusion pip
conda activate MuseDiffusion
pip3 install -r requirements.txt
```

<h3>2. Preprocess dataset</h3>

```bash
python3 -m MuseDiffusion.data --num_proc <num-proc>
```
* where `<num-proc>` can be optimized, according to your node spec.

<h4>Directory Structure</h4>

After this step, your directory structure would be like:

```
MuseDiffusion
├── MuseDiffusion
│   ├── __init__.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── base.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── corruption.py
│   │   └── ...
│   ├── models
│   │   ├── __init__.py
│   │   ├── commu
│   │   │   └── (ComMU python module files...)
│   │   └── diffusion
│   │       ├── __init__.py
│   │       ├── classifier.py
│   │       ├── denoising_model.py
│   │       └── ...
│   ├── run
│   │   ├── __init__.py
│   │   ├── sample_generation.py
│   │   ├── sample_seq2seq.py
│   │   └── train.py
│   └── utils
│       ├── __init__.py
│       ├── decode_util.py
│       ├── dist_util.py
│       └── ...
├── assets
│   └── (files for readme...)
├── datasets
│   └── ComMU-processed
│       └── (preprocessed commu dataset files...)
├── scripts
│   ├── run_generation.sh
│   ├── run_seq2seq.sh
│   └── run_train.sh
├── README.md
├── pozalabs_embedding.pt
└── requirements.txt
```

<h3>3. Prepare model weight and configuration</h3>

<h4>With downloading pretrained one</h4>

```bash
mkdir diffusion_models
cd diffusion_models
curl -fsSL -o pretrained_weights.zip <URL_TBD>
unzip pretrained_weights.zip && rm pretrained_weights.zip
cd ..
```

<h4>With Manual Training</h4>

```bash
# Copy config file to root directory
python3 -m MuseDiffusion.config --save_into train_cfg.json

# Optional: customize config on your own
vi train_cfg.json

# Run training script
python3 -m MuseDiffusion.run.train --distributed --config_json train_cfg.json
```
* Note: required arguments will be automatically loaded from `train_cfg.json`. \
  if you want not to use json config, you could manually type arguments, \
  refer to signatures in `python3 -m MuseDiffusion.run.train --help`.
* Note: argument `--distributed` will run `MuseDiffusion.run.train` \
  **with torch.distributed runner**. (you can omit this argument.) \
* Note: you can customize distributed options e.g. `--nproc_per_node`, `--master_port`, \
  or environs e.g. `OPT_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`. \
  (`OPT_NUM_THREADS` will be set to `$CPU_CORE` / / `$TOTAL_GPU` in default.)
* Note: In windows, torch.distributed is disabled in default. \
  to enable, set `MuseDiffusion.utils.dist_util.USE_DIST_IN_WINDOWS` to `True`.
* Note: After training, weights and configs will be saved into `./diffusion_models/<name>/`. \
  **<u>do not move or delete ANY FILE ENDS WITH .PT OR .JSON in this directory</u>**.

<h3>4. Sample with model!</h3>

<h4>From corrupted samples</h4>

```bash
$ TBD
```

<h4>From metadata</h4>

```bash
$ TBD
```
<br>
<hr>
<h2> Datasets</h2>


<br>
<br>
<img src="https://i.ibb.co/8c9Scmt/2.png">
