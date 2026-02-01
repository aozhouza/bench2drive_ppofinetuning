<img width="172" height="32" alt="image" src="https://github.com/user-attachments/assets/5011f179-8f38-4f46-ad84-7bf6c87c1e37" />## Install
### Create a conda environment and install
```
conda create -n b2d_ppo_finetuning python=3.8
conda activate b2d_ppo_finetuning
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# cuda 11.8 and GCC 9.4 is strongly recommended. Otherwise, it might encounter errors.
export PATH=YOUR_GCC_PATH/bin:$PATH
export CUDA_HOME=YOUR_CUDA_PATH/
pip install ninja packaging
cd Bench2DriveZoo-uniad-vad
pip install -v -e .
```
### Prepare pretrained weights.
```
cd Bench2DriveZoo-uniad-vad
mkdir ckpts
```
Download `resnet50-19c8e357.pth` from [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth).

Download `r101_dcn_fcos3d_pretrain.pth` from [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth).

Download `uniad_base_b2d.pth` from [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth).

Download `vad_b2d_base.pth` from [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/vad_b2d_base.pth).
### Install CARLA for closed-loop ppo fine-tuning.
```
cd ..
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
export CARLA_ROOT=YOUR_CARLA_PATH
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.8/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
```
### Closed Loop PPO fine-tuning.
```
cd Bench2Drive-main/leaderboard
mkdir team_code
ln -s Bench2DriveZoo-uniad-vad/team_code/* ./team_code
cd ..
ln -s Bench2DriveZoo-uniad-vad  ./ 
```
The core code for closed-loop fine-tuning of PPO is located in `Bench2DriveZoo-uniad-vad/team_code`.
PPO fine-tuning of training closed loop.
```
cd Bench2Drive-main
bash leaderboard/scripts/run_evaluation.sh
```
### Metric Calculation
```
# Composite Driving Performance Score Calculation
cd Bench2Drive-main
python cal_res.py
# Success Rate Calculation
cd Bench2Drive-main/tools
python merge_route_json.py
# Efficiency and Comfortness Calculation
python efficiency_smoothness_benchmark.py
```
If Carla crashes, run the following command to clear it.
```
bash Bench2Drive-main/tools/clean_carla.sh
```

