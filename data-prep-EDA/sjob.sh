#!/bin/bash 
 
#SBATCH --job-name=recsys
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j_%N.log 

#SBATCH --mem=512gb  
#SBATCH --cpus-per-task=128
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=batch

source /home/tphasit/PxC-book-recommendation/data-prep-EDA/data/bin/activate
python clean_tags.py