<img src="https://i.ibb.co/z66nz7q/1.png">
<br>
<h2> How to run</h2>

<h3>0. Clone repository and cd</h3>

```bash
git clone https://github.com/YAIxPOZAlabs/<repository-name>.git
cd <repository-name>
```

<h3>1. Prepare environment and data</h3>

<h4>(Optional) Install python 3.8 <i>for Virtualenv usage</i></h4>

```bash
sudo apt update && \
sudo apt install -y software-properties-common && \
sudo add-apt-repository -y ppa:deadsnakes/ppa && \
sudo apt install -y python3.8 python3.8-distutils
```

<h4>With Virtualenv & PyPI</h4>

```bash
python3 -m pip install virtualenv && \
python3 -m virtualenv venv --python=python3.8 && \
source venv/bin/activate && \  # 
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
pip3 install -r requirements.txt
```

<h4>With Anaconda</h4>

```bash
conda env create --file environment.yaml && \
conda activate <environment-name>
```

<h3>2. Preprocess dataset</h3>

```bash
python3 -m data --num_proc <num-proc>
```
* where `<num-proc>` can be optimized, according to your node spec.

#### <<<<<TODO: directory structure>>>>>

<h3>3. Prepare model weight and configuration</h3>

<h4>With downloading pretrained one</h4>

```bash
$ TBD
```

<h4>With Manual Training</h4>

```bash
# Copy config file to root directory
python3 -m config --save_into train_cfg.json

# Optional: customize config on your own
vi train_cfg.json

# Run training script
python3 scripts/run_train.py --config_file train_cfg.json --notes commu
```
* Note: required arguments of `train.py` will be automatically loaded from `train_cfg.json`.
* Note: script `scripts/run_train.py` will **run `train.py` with torch.distributed runner**. \
  you can customize distributed options e.g. `--nproc_per_node`, `--master_port`, \
  or environs e.g. `OPT_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`. \
  (tip: set `OPT_NUM_THREADS` to `$CPU_CORE` / / `$TOTAL_GPU`)
* Note: In windows, torch.distributed is disabled in default. \
  to enable, set `utils.dist_util.USE_DIST_IN_WINDOWS` to `True`.
* Note: After training, **weights and configs will be saved into `./diffusion_models/<name>/`**. \
  **<u>do not move or delete ANY FILE ENDS WITH .PT OR .JSON in this directory</u>**.

<br>
<br>
<img src="https://i.ibb.co/8c9Scmt/2.png">
