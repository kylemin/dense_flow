#!/bin/bash
#SBATCH --job-name=df
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --exclude=lens,eyeball,focus,cmos,macula,mirror,rhodopsin,fovea,zonule
#SBATCH --time=5000  # Time in Minutes
#SBATCH --output=./oefiles/%j-%N.out
#SBATCH --mail-user=kylemin@umich.edu
#SBATCH --mail-type=ALL

hostname
nvidia-smi
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
unset PYTHONPATH
module load cuda/9.0 cudnn/9.0-v7.1.2 ffmpeg libgpuarray
export PYTHONPATH=/z/home/kylemin/library/python2-libs:$PYTHONPATH
export OpenCV_DIR=/z/home/kylemin/library/opencv-4.1.1-python2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OpenCV_DIR/lib
export PYTHONPATH=$PYTHONPATH:$OpenCV_DIR/lib/python2.7/dist-packages
export PATH=$PATH:$OpenCV_DIR/bin
cd /z/home/kylemin/library/dense_flow

python2 extract_flow_h.py
