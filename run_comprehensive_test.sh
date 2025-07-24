#!/bin/bash

# =============================================================================
# Comprehensive Metric Generation Test Launcher
# =============================================================================
# This script helps launch the comprehensive metric generation test suite
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

display_banner() {
    echo ""
    echo "=============================================="
    echo "🚀 Comprehensive Metric Generation Test Suite"
    echo "=============================================="
    echo ""
    echo "This test suite will:"
    echo "  ✓ Test all 8 metric generation types"
    echo "  ✓ Validate serialization and reloading"
    echo "  ✓ Verify metric card generation"
    echo "  ✓ Test Prometheus and DSPy integration"
    echo "  ✓ Ensure rock-solid reliability"
    echo ""
    echo "Metric types tested:"
    echo "  1. Basic LLM Judge"
    echo "  2. LLM Judge with Example Selection"
    echo "  3. LLM Judge with MIPROv2 Optimization"
    echo "  4. G-Eval"
    echo "  5. Code Generation"
    echo "  6. Rubric Generator with Prometheus"
    echo "  7. Rubric Generator with DSPy"
    echo "  8. Fine-tuned ModernBERT"
    echo ""
    echo "=============================================="
    echo ""
}

check_environment() {
    log_info "Checking environment..."
    
    # Check if we're in the right directory
    if [ ! -f "metric_generation_demo.py" ]; then
        log_error "Please run this from the autometrics root directory"
        exit 1
    fi
    
    # Check if test scripts exist
    if [ ! -f "test_all_metric_generation_types.sh" ]; then
        log_error "test_all_metric_generation_types.sh not found!"
        exit 1
    fi
    
    if [ ! -f "submit_comprehensive_metric_test.sh" ]; then
        log_error "submit_comprehensive_metric_test.sh not found!"
        exit 1
    fi
    
    # Check if scripts are executable
    if [ ! -x "test_all_metric_generation_types.sh" ]; then
        log_warning "Making test_all_metric_generation_types.sh executable..."
        chmod +x test_all_metric_generation_types.sh
    fi
    
    if [ ! -x "submit_comprehensive_metric_test.sh" ]; then
        log_warning "Making submit_comprehensive_metric_test.sh executable..."
        chmod +x submit_comprehensive_metric_test.sh
    fi
    
    log_success "Environment check passed"
}

show_options() {
    echo "Choose how to run the comprehensive test:"
    echo ""
    echo "1. 🖥️  Run locally (requires GPU and may take 2-6 hours)"
    echo "2. 🚀 Submit to SLURM cluster (recommended)"
    echo "3. 📋 Show test configuration"
    echo "4. 📖 Show detailed information"
    echo "5. 🚪 Exit"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            run_local
            ;;
        2)
            submit_to_slurm
            ;;
        3)
            show_configuration
            ;;
        4)
            show_detailed_info
            ;;
        5)
            log_info "Exiting..."
            exit 0
            ;;
        *)
            log_error "Invalid choice. Please enter 1-5."
            show_options
            ;;
    esac
}

run_local() {
    log_warning "Running locally requires:"
    log_warning "  - A GPU with sufficient memory"
    log_warning "  - 2-6 hours of runtime"
    log_warning "  - Stable network connection"
    echo ""
    read -p "Are you sure you want to run locally? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        log_info "Starting comprehensive test locally..."
        ./test_all_metric_generation_types.sh
    else
        log_info "Returning to main menu..."
        show_options
    fi
}

submit_to_slurm() {
    log_info "Submitting comprehensive test to SLURM cluster..."
    
    # Show job details
    echo ""
    echo "Job configuration:"
    echo "  - Time limit: 6 hours"
    echo "  - Memory: 100GB"
    echo "  - CPUs: 16"
    echo "  - GPUs: 1"
    echo "  - Partition: sc-loprio"
    echo ""
    
    # Submit the job
    if sbatch submit_comprehensive_metric_test.sh; then
        log_success "Job submitted successfully!"
        echo ""
        log_info "Monitor job status with: squeue -u $USER"
        log_info "Cancel job with: scancel <job_id>"
        log_info "View logs in: test_logs/"
        echo ""
        log_info "The test will run all 8 metric types and validate serialization."
        log_info "You'll receive an email when the job completes."
    else
        log_error "Failed to submit job to SLURM"
    fi
}

show_configuration() {
    log_info "Test Configuration:"
    echo ""
    echo "🔧 Servers:"
    echo "  - Qwen: localhost:7450 (will be started automatically)"
    echo "  - Prometheus: future-hgx-1:7420 (should be running)"
    echo ""
    echo "📊 Test Parameters:"
    echo "  - Seed: 42 (for reproducibility)"
    echo "  - Metrics per test: 2 (except finetune: 1)"
    echo "  - Timeout: 2 hours"
    echo ""
    echo "📁 Directories:"
    echo "  - Cache: /nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_test"
    echo "  - Models: /sphinx/u/salt-checkpoints/autometrics/models"
    echo "  - Output: ./generated_metrics/"
    echo "  - Logs: ./test_logs/"
    echo ""
    echo "🧪 What gets tested:"
    echo "  - Metric generation for all 8 types"
    echo "  - Serialization to Python files"
    echo "  - Reloading and execution verification"
    echo "  - Metric card generation"
    echo "  - Reference-based vs reference-free detection"
    echo ""
    read -p "Press Enter to return to main menu..."
    show_options
}

show_detailed_info() {
    log_info "Detailed Test Information:"
    echo ""
    echo "📋 Test Process:"
    echo "  1. Environment setup and prerequisite checks"
    echo "  2. Qwen server startup (Qwen3-32B model)"
    echo "  3. Prometheus server connectivity check"
    echo "  4. Sequential testing of all 8 metric types"
    echo "  5. Serialization validation for each metric"
    echo "  6. Server cleanup and result summary"
    echo ""
    echo "⏱️ Expected Runtime:"
    echo "  - Basic LLM Judge: 10-15 minutes"
    echo "  - LLM Judge Examples: 15-20 minutes"
    echo "  - LLM Judge Optimized: 20-30 minutes"
    echo "  - G-Eval: 10-15 minutes"
    echo "  - Code Generation: 5-10 minutes"
    echo "  - Rubric Prometheus: 15-20 minutes"
    echo "  - Rubric DSPy: 10-15 minutes"
    echo "  - Fine-tuned ModernBERT: 30-60 minutes"
    echo "  - Total: 2-6 hours"
    echo ""
    echo "🔍 What gets validated:"
    echo "  - Metric generation completes without errors"
    echo "  - Generated metrics are serialized to Python files"
    echo "  - Serialized metrics can be loaded and executed"
    echo "  - Metric scores are consistent between runs"
    echo "  - Metric cards are generated properly"
    echo "  - Reference-based vs reference-free detection works"
    echo ""
    echo "📊 Success Criteria:"
    echo "  - All 8 metric types pass generation"
    echo "  - All generated metrics serialize correctly"
    echo "  - All serialized metrics reload and execute"
    echo "  - No critical errors in any test phase"
    echo ""
    read -p "Press Enter to return to main menu..."
    show_options
}

main() {
    display_banner
    check_environment
    show_options
}

# Run the main function
main "$@" 