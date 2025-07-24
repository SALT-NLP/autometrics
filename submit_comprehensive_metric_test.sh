#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --job-name=comprehensive_metric_test
#SBATCH --output=test_logs/comprehensive_metric_test.out
#SBATCH --error=test_logs/comprehensive_metric_test.err
#SBATCH --constraint=141G
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# =============================================================================
# SLURM Job Script for Comprehensive Metric Generation Testing
# =============================================================================
# This script submits the comprehensive metric generation test to the cluster
# with appropriate hardware resources for running Qwen server and all tests
# =============================================================================

set -euo pipefail

# Handle SLURM variables that might not be set
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-100000}
export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS:-1}}
export SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT:-6:00:00}

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

# =============================================================================
# Job Information
# =============================================================================

log_info "Starting comprehensive metric generation test job"
log_info "Job ID: $SLURM_JOB_ID"
log_info "Node: $SLURM_NODELIST"
log_info "Start Time: $(date)"
log_info "Working Directory: $(pwd)"

# =============================================================================
# Environment Setup
# =============================================================================

log_info "Setting up environment..."

# Load conda
source /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

# Create log directory
mkdir -p test_logs

# Change to the autometrics directory
cd /nlp/scr2/nlp/personal-rm/autometrics

# Set OpenAI API key if available
if [ -n "${OPENAI_API_KEY:-}" ]; then
    log_info "OpenAI API key is set"
else
    log_info "OpenAI API key not set - some metric card generation may be limited"
fi

# =============================================================================
# Pre-flight Checks
# =============================================================================

log_info "Running pre-flight checks..."

# Check if test script exists
if [ ! -f "test_all_metric_generation_types.sh" ]; then
    log_error "test_all_metric_generation_types.sh not found!"
    exit 1
fi

# Check if main demo script exists
if [ ! -f "metric_generation_demo.py" ]; then
    log_error "metric_generation_demo.py not found!"
    exit 1
fi

# Check conda environments
if ! conda info --envs | grep -q "autometrics"; then
    log_error "autometrics conda environment not found"
    exit 1
fi

if ! conda info --envs | grep -q "sglang"; then
    log_error "sglang conda environment not found"
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
    log_info "nvidia-smi not available"
fi

# =============================================================================
# Run the Comprehensive Test
# =============================================================================

log_info "Launching comprehensive metric generation test..."
log_info "This will test all 8 metric types with serialization and reloading"

# Run the main test script
if ./test_all_metric_generation_types.sh; then
    log_success "Comprehensive metric generation test completed successfully!"
    
    # Show summary of generated files
    log_info "Generated metric files:"
    find generated_metrics -name "*.py" -type f | head -10
    
    # Show log files
    log_info "Test log files:"
    ls -la test_logs/*.log | head -10
    
    exit 0
else
    log_error "Comprehensive metric generation test failed!"
    
    # Show recent error logs
    log_info "Recent error logs:"
    for log_file in test_logs/*.log; do
        if [ -f "$log_file" ]; then
            echo "=== Last 10 lines of $log_file ==="
            tail -n 10 "$log_file"
            echo ""
        fi
    done
    
    exit 1
fi

# =============================================================================
# Job Completion
# =============================================================================

log_info "Job completed at: $(date)"
log_info "Total runtime: $(date -d@$SECONDS -u +%H:%M:%S)"

# Final status
if [ $? -eq 0 ]; then
    log_success "🎉 ALL METRIC GENERATION TYPES TESTED SUCCESSFULLY!"
    log_success "The autometrics metric generation system is rock solid!"
else
    log_error "Some tests failed. Check the logs for details."
fi 