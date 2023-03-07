<!-- HEADER START -->
<!-- src: https://github.com/kyechan99/capsule-render -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:B993D6,100:8CA6DB&height=220&section=header&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=40&text=%E2%97%A6%20%CB%9A%20%EF%BC%B9%EF%BC%A1%EF%BC%A9%20%C3%97%20%EF%BC%B0%EF%BD%8F%EF%BD%9A%EF%BD%81%EF%BD%8C%EF%BD%81%EF%BD%82%EF%BD%93%20%CB%9A%20%E2%97%A6" alt="header" />
</a></p>
<h3 align="center">Midi-data Modification based on Diffusion Model</h3>
<p align="center"><a href="https://github.com/YAIxPOZAlabs"><img src="assets/logo.png" width="50%" height="50%" alt="logo"></a></p>
<p align="center">This project was carried out by <b><a href="https://github.com/yonsei-YAI">YAI 11th</a></b>, in cooperation with <b><a href="https://github.com/POZAlabs">POZAlabs</a></b>.</p>
<p align="center">
<br>
<a href="mailto:dhakim@yonsei.ac.kr">
    <img src="https://img.shields.io/badge/-Gmail-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Gmail"/>
</a> 
<a href="https://www.notion.so/dhakim/1e7dc19fd1064e698a389f75404883c7?pvs=4">
    <img src="https://img.shields.io/badge/-Notion-000000?style=flat-square&logo=notion&logoColor=white" alt="NOTION"/>
</a> 
</p>
<br>

---

<!-- HEADER END -->

<h3 align="center"><br>✨&nbsp; Contributors&nbsp; ✨<br><br></h3>
<p align="center">
<b>🛠️ <a href="https://github.com/kdha0727">KIM DONGHA</a></b>&nbsp; :&nbsp; YAI 8th&nbsp; /&nbsp; AI Dev Lead &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
<b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🚀 <a href="https://github.com/ta3h30nk1m">KIM TAEHEON</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp; AI Research & Dev <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>👑 <a href="https://github.com/san9min">LEE SANGMIN</a></b>&nbsp; :&nbsp; YAI 9th&nbsp; /&nbsp; Executive Director <br>
&nbsp;<b>🐋 <a href="https://github.com/Tim3s">LEE SEUNGJAE</a></b>&nbsp; :&nbsp; YAI 9th&nbsp; /&nbsp; AI Research Lead <br>
<b>🌈 <a href="https://github.com/jeongwoo1213">CHOI JEONGWOO</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp; AI Research & Dev <br>
<b>🌟 <a href="https://github.com/starwh03">CHOI WOOHYEON</a></b>&nbsp; :&nbsp; YAI 10th&nbsp; /&nbsp; AI Research & Dev <br>
<br><br>
<hr>
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

<h3>2. Download and Preprocess dataset</h3>

```bash
python3 -m MuseDiffusion.data [--num_proc {NUM_PROC}]
```
* where `{NUM_PROC}` can be optimized, according to your node spec.

<h4>Directory Structure</h4>

After this step, your directory structure would be like:

```
MuseDiffusion
├── commu
│   └── (https://github.com/POZAlabs/ComMU-code/blob/master/commu/)
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
│   │   ├── denoising_model.py
│   │   ├── gaussian_diffusion.py
│   │   ├── nn.py
│   │   └── ...
│   ├── run
│   │   ├── __init__.py
│   │   ├── sample_generation.py
│   │   ├── sample_seq2seq.py
│   │   └── train.py
│   └── utils
│       ├── __init__.py
│       ├── decode_util.py
│       ├── dist_util.py
│       ├── train_util.py
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
* Note: required arguments will be automatically loaded from `train_cfg.json`.
  if you want not to use json config, you could manually type arguments,
  refer to signatures in `python3 -m MuseDiffusion.run.train --help`.
* Note: argument `--distributed` will run `MuseDiffusion.run.train` 
  **with torch.distributed runner**, and you can customize options, or environs.
  * commandline option `--nproc_per_node` - number of training node (GPU) to use. \
    (default: number of GPU in `CUDA_VISIBLE_DEVICES` environ.)
  * commandline option `--master_port` - master port for distributed learning. \
    (you should specify it for multiple runs.)
  * environ `CUDA_VISIBLE_DEVICES` - specific GPU index. e.g: `CUDA_VISIBLE_DEVICES=4,5,6,7` \
    (default: not set - in this case, trainer will use all available GPUs.)
  * environ `OPT_NUM_THREADS` - number of threads for each node. \
    (default: `$CPU_CORE` / / `$TOTAL_GPU`)
* Note: In windows, torch.distributed is disabled in default. 
  to enable, edit `MuseDiffusion.utils.dist_util.USE_DIST_IN_WINDOWS`.
* Note: After training, weights and configs will be saved into `./diffusion_models/{name-of-model-folder}/`. \
  <u>**do not move or delete ANY FILE ENDS WITH .PT OR .JSON in this directory**</u>.


<h3>4. Sample with model - Modify or Generate Midi!</h3>

<h4>From corrupted samples</h4>

```bash
python3 -m MuseDiffusion.run.sample modification --distributed \
--use_corruption True \
--corr_available mt,mn,rn,rr \
--corr_max 4 \
--corr_p 0.5 \
--model_path diffusion_models/{name-of-model-folder}/{weight-file} \
--step 1000 \
--top_p 1 \
--clamp_step 0 \
--clip_denoised true \
--sample_seed 123
```
* Note: You can use arguments for `torch.distributed`, which is same as training script.
* Note: Type `python3 -m MuseDiffusion.run.sample modification --help` for detailed usage.

<h4>From metadata</h4>

```bash
python3 -m MuseDiffusion.run.sample generation --distributed \
--bpm {BPM} \
--audio_key {AUDIO_KEY} \
--time_signature {TIME_SIGNATURE} \
--pitch_range {PITCH_RANGE} \
--num_measures {NUM_MEASURES} \
--inst {INST} \
--genre {GENRE} \
--min_velocity {MIN_VELOCITY} \
--max_velocity {MAX_VELOCITY} \
--track_role {TRACK_ROLE} \
--rhythm {RHYTHM} \
--chord_progression {CHORD_PROGRESSION} \
--num_samples 1000 \
--model_path diffusion_models/{name-of-model-folder}/{weight-file} \
--step 1000 \
--top_p 1 \
--clamp_step 0 \
--clip_denoised true \
--sample_seed 123
```
* Note: You can use arguments for `torch.distributed`, which is same as training script.
* Note: Type `python3 -m MuseDiffusion.run.sample generation --help` for detailed usage.

<br>

---

## Datasets

<h3 align="center">ComMU: Dataset for Combinatorial Music Generation</h3>

> Combinatorial music generation creates short samples of music with rich musical metadata, 
> and combines them to produce a complete music. 
> <u>**ComMU**</u> is the first symbolic music dataset consisting of short music samples 
> and their corresponding 12 musical metadata for combinatorial music generation. 
> 
> Notable properties of ComMU are that (1) dataset is manually constructed by professional composers 
> with an objective guideline that induces regularity, 
> and (2) it has 12 musical metadata that embraces composers' intentions. 
> 
> ComMU's results show that we can generate diverse high-quality music only with metadata, and that 
> our unique metadata such as track-role and extended chord quality improves the capacity of the automatic composition.

---

<!-- FOOTER START -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:8CA6DB,100:B993D6&height=180&section=footer&animation=fadeIn&fontAlignY=40" alt="header" />
</a></p>
<!-- FOOTER END -->
