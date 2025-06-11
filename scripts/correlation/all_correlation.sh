#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --job-name=all_correlation
#SBATCH --output=logs/all_correlation.out
#SBATCH --error=logs/all_correlation.err
#SBATCH --constraint=141G
#SBATCH --requeue

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymLessonOutcome --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymLessonProcess --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTabularOutcome --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTabularProcess --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelOutcome --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset CoGymTravelProcess --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset Design2Code --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset EvalGenMedical --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset EvalGenProduct --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset HelpSteer --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset HelpSteer2 --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset Primock57 --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset RealHumanEval --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset SimpDA --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset SimpEval --correlation all --top-k 5
python autometrics/experiments/correlation/benchmark_correlation.py --dataset SummEval --correlation all --top-k 5