## Install
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
Download `resnet50-19c8e357.pth` from [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth)

Download `r101_dcn_fcos3d_pretrain.pth` from [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth)

