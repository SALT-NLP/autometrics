#!/bin/bash

#SBATCH --job-name=alpha_test_simpda
#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --partition=jag-lo
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --gres=gpu:4
#SBATCH -x jagupard[19-20,26-31]
#SBATCH --output=alpha_test_simpda.out
#SBATCH --error=alpha_test_simpda.err

# =============================================================================
# SLURM Job Script for Alpha Comparison Test - SimpDA Dataset
# =============================================================================

# Load conda FIRST (before any error handling)
. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

cd /nlp/scr2/nlp/personal-rm/autometrics

# Now set error handling after conda is loaded
set -euo pipefail

# Handle SLURM variables that might not be set (provide defaults to prevent unbound variable errors)
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-100000}
export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
export SLURM_GPUS=${SLURM_GPUS:-1}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-48:00:00}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[SLURM INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SLURM SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[SLURM ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[SLURM WARNING]${NC} $1"; }

# =============================================================================
# Job Information
# =============================================================================

log_info "Starting alpha comparison test for SimpDA dataset"
log_info "Job ID: $SLURM_JOB_ID"
log_info "Node: $SLURM_NODELIST"
log_info "Start Time: $(date)"
log_info "Working Directory: $(pwd)"

# =============================================================================
# Environment Setup
# =============================================================================

log_info "Setting up environment..."

# =============================================================================
# Pre-flight Checks
# =============================================================================

log_info "Running pre-flight checks..."

# Check if test script exists
if [ ! -f "test_regression_alpha_comparison.py" ]; then
    log_error "test_regression_alpha_comparison.py not found!"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "autometrics/__init__.py" ]; then
    log_error "autometrics package not found in current directory"
    exit 1
fi

log_success "Pre-flight checks passed"

# =============================================================================
# Resource Information
# =============================================================================

log_info "Resource allocation:"
log_info "  - CPUs: $SLURM_CPUS_PER_TASK"
log_info "  - Memory: $SLURM_MEM_PER_NODE MB"
log_info "  - GPUs: $SLURM_GPUS_PER_NODE"
log_info "  - Time limit: $SLURM_TIME_LIMIT"

# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    log_warning "nvidia-smi not available"
fi

# =============================================================================
# Run the Alpha Comparison Test for SimpDA
# =============================================================================

log_info "Launching alpha comparison test for SimpDA dataset..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Set dataset-specific DSPY cache directory
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_alpha_test_simpda"
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

# Parse command line arguments (optional)
SEED=${1:-42}
REGENERATE=${2:-""}

log_info "Test configuration:"
log_info "  - Dataset: SimpDA"
log_info "  - Target Measure: simplicity (auto-detected)"
log_info "  - Seed: $SEED"
log_info "  - Regenerate Metrics: $REGENERATE"
log_info "  - Generated Metrics Dir: generated_metrics_SimpDA"
log_info "  - DSPY Cache Dir: $DSPY_CACHEDIR"

# Build command
CMD="COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True python test_regression_alpha_comparison.py --dataset SimpDA --target-measure simplicity --seed $SEED"
if [ "$REGENERATE" = "true" ]; then
    CMD="$CMD --regenerate-metrics"
fi

log_info "Running command: $CMD"

# Run the test script
if eval $CMD; then
    log_success "Alpha comparison test for SimpDA completed successfully!"
    
    # Show output file size
    if [ -f "alpha_test_simpda.out" ]; then
        log_info "Output file size: $(du -h alpha_test_simpda.out | cut -f1)"
    fi
    
    exit 0
else
    log_error "Alpha comparison test for SimpDA failed!"
    
    # Show recent error output
    if [ -f "alpha_test_simpda.err" ]; then
        log_info "Error output (last 20 lines):"
        tail -n 20 alpha_test_simpda.err
    fi
    
    exit 1
fi

# =============================================================================
# Job Completion
# =============================================================================

log_info "Job completed at: $(date)"

# Final status
if [ $? -eq 0 ]; then
    log_success "🎉 ALPHA COMPARISON TEST FOR SIMPDA COMPLETED SUCCESSFULLY!"
else
    log_error "Test failed. Check the logs for details."
fi 