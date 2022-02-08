#!/bin/sh
#SBATCH --job-name=pbp_api_call
#SBATCH --partition=dev_single
#SBATCH --mem=16000
#SBATCH --output=name
#SBATCH --error=name
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=finn.hoener@student.uni-tuebingen.de

# path to where conda is and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsp

# source the python file, optionally add arguments behind
python  pbp_api_call.py
