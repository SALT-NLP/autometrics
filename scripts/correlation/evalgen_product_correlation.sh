#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --job-name=evalgen_product_correlation
#SBATCH --output=logs/evalgen_product_correlation.out
#SBATCH --error=logs/evalgen_product_correlation.err
#SBATCH --constraint=141G

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

python autometrics/experiments/correlation/benchmark_correlation.py --dataset EvalGenProduct --correlation all --top-k 5