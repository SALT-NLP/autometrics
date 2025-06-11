#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=jag-lo
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --job-name=cogym_travel_outcome
#SBATCH --output=logs/cogym_travel_outcome.out
#SBATCH --error=logs/cogym_travel_outcome.err
#SBATCH --exclude=jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelOutcome --correlation all --top-k 5