#!/bin/bash

# =============================================================================
# Comprehensive Metric Generation Testing Script
# =============================================================================
# This script tests all metric generation types end-to-end with serialization
# to ensure rock-solid functionality across the complete feature set.
#
# USAGE:
#   To run all tests:
#     ./test_all_metric_generation_types.sh
#
#   To test only specific metric types, edit the ENABLE_* variables below.
#   For example, to test only the problematic metrics:
#     ENABLE_LLM_JUDGE=0
#     ENABLE_GEVAL=0
#     ENABLE_LLM_JUDGE_EXAMPLES=1
#     ENABLE_LLM_JUDGE_OPTIMIZED=1
#     ENABLE_CODEGEN=1
#     ENABLE_FINETUNE=1
#     ENABLE_RUBRIC_PROMETHEUS=0
#     ENABLE_RUBRIC_DSPY=0
# =============================================================================

set -euo pipefail  # Exit on error, undefined variables, pipe failures

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

# =============================================================================
# Configuration
# =============================================================================

# Server configuration
QWEN_MODEL="Qwen/Qwen3-32B"
QWEN_PORT=7440  # Different port to avoid conflicts
PROMETHEUS_SERVER="http://future-hgx-1:7420/v1"
QWEN_API_BASE="http://localhost:${QWEN_PORT}/v1"

# Test configuration
TEST_SEED=42
N_METRICS=2  # Keep low for faster testing but still comprehensive
TIMEOUT_MINUTES=120  # 2 hours timeout

# Environment configuration
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_test"
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

# Ensure directories exist
mkdir -p "$DSPY_CACHEDIR"
mkdir -p "$AUTOMETRICS_MODEL_DIR"
mkdir -p "generated_metrics"
mkdir -p "test_logs"

# =============================================================================
# Metric Type Toggles - Enable/Disable specific tests for faster iteration
# =============================================================================

# Set to 1 to enable, 0 to disable specific metric types
ENABLE_LLM_JUDGE=1
ENABLE_LLM_JUDGE_EXAMPLES=1
ENABLE_LLM_JUDGE_OPTIMIZED=1
ENABLE_GEVAL=1
ENABLE_CODEGEN=1
ENABLE_RUBRIC_PROMETHEUS=1
ENABLE_RUBRIC_DSPY=1
ENABLE_FINETUNE=1

# =============================================================================
# Metric Types to Test
# =============================================================================

# Build the list of metric types based on enabled toggles
declare -a METRIC_TYPES=()

if [ "$ENABLE_LLM_JUDGE" -eq 1 ]; then
    METRIC_TYPES+=("llm_judge:Basic LLM Judge")
fi

if [ "$ENABLE_LLM_JUDGE_EXAMPLES" -eq 1 ]; then
    METRIC_TYPES+=("llm_judge_examples:LLM Judge with Example Selection")
fi

if [ "$ENABLE_LLM_JUDGE_OPTIMIZED" -eq 1 ]; then
    METRIC_TYPES+=("llm_judge_optimized:LLM Judge with MIPROv2 Optimization")
fi

if [ "$ENABLE_GEVAL" -eq 1 ]; then
    METRIC_TYPES+=("geval:G-Eval")
fi

if [ "$ENABLE_CODEGEN" -eq 1 ]; then
    METRIC_TYPES+=("codegen:Code Generation")
fi

if [ "$ENABLE_RUBRIC_PROMETHEUS" -eq 1 ]; then
    METRIC_TYPES+=("rubric_prometheus:Rubric Generator with Prometheus")
fi

if [ "$ENABLE_RUBRIC_DSPY" -eq 1 ]; then
    METRIC_TYPES+=("rubric_dspy:Rubric Generator with DSPy")
fi

if [ "$ENABLE_FINETUNE" -eq 1 ]; then
    METRIC_TYPES+=("finetune:Fine-tuned ModernBERT")
fi

# Verify we have at least one metric type enabled
if [ ${#METRIC_TYPES[@]} -eq 0 ]; then
    log_error "No metric types enabled! Please set at least one ENABLE_* variable to 1."
    exit 1
fi

log_info "Enabled metric types: ${#METRIC_TYPES[@]}"
for metric_info in "${METRIC_TYPES[@]}"; do
    description="${metric_info#*:}"
    log_info "  ✓ $description"
done

# =============================================================================
# Helper Functions
# =============================================================================

cleanup() {
    log_info "Cleaning up servers and processes..."
    
    # Kill Qwen server if running
    if pgrep -f "sglang.launch_server.*${QWEN_PORT}" > /dev/null; then
        log_info "Stopping Qwen server on port ${QWEN_PORT}..."
        pkill -f "sglang.launch_server.*${QWEN_PORT}" || true
        sleep 5
    fi
    
    # Additional cleanup for any hanging processes
    pkill -f "python.*metric_generation_demo.py" || true
    
    log_info "Cleanup completed"
}

# Set up cleanup trap
trap cleanup EXIT

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "metric_generation_demo.py" ]; then
        log_error "Please run this script from the autometrics root directory"
        exit 1
    fi
    
    # Check conda environment
    if ! conda info --envs | grep -q "autometrics"; then
        log_error "autometrics conda environment not found"
        exit 1
    fi
    
    # Check if Prometheus server is accessible
    if ! curl -s "${PROMETHEUS_SERVER}/get_model_info" > /dev/null; then
        log_warning "Prometheus server at ${PROMETHEUS_SERVER} may not be accessible"
        log_warning "rubric_prometheus tests may fail"
    else
        log_success "Prometheus server is accessible"
    fi
    
    # Check OpenAI API key (optional but recommended)
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        log_warning "OPENAI_API_KEY not set - metric card generation may be limited"
    else
        log_success "OpenAI API key is set"
    fi
    
    log_success "Prerequisites check completed"
}

start_qwen_server() {
    log_info "Starting Qwen server on port ${QWEN_PORT}..."
    
    # Check if server is already running
    if curl -s "${QWEN_API_BASE}/v1/models" > /dev/null; then
        log_success "Qwen server is already running on port ${QWEN_PORT}"
        return 0
    fi
    
    # Switch to sglang environment for server startup
    source /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
    conda activate sglang
    
    # Start the server in background
    log_info "Launching Qwen server: ${QWEN_MODEL}..."
    python -m sglang.launch_server \
        --model-path "${QWEN_MODEL}" \
        --port "${QWEN_PORT}" \
        --host 0.0.0.0 \
        --tp 1 \
        --dtype bfloat16 \
        --mem-fraction-static 0.8 \
        --trust-remote-code \
        > "test_logs/qwen_server_${QWEN_PORT}.log" 2>&1 &
    
    # Wait for server to be ready
    log_info "Waiting for Qwen server to be ready..."
    local start_time=$(date +%s)
    local timeout=$((TIMEOUT_MINUTES * 60))
    
    while ! curl -s "${QWEN_API_BASE}/v1/models" > /dev/null; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            log_error "Timeout reached after ${TIMEOUT_MINUTES} minutes. Server startup failed."
            exit 1
        fi
        
        echo -n "."
        sleep 10
    done
    
    echo ""
    log_success "Qwen server is ready on port ${QWEN_PORT}!"
    
    # Switch back to autometrics environment
    conda activate autometrics
}

test_metric_type() {
    local metric_type="$1"
    local description="$2"
    
    log_info "Testing ${description} (${metric_type})..."
    log_info "  Parameters: n_metrics=${N_METRICS}, seed=${TEST_SEED}"
    log_info "  API base: ${QWEN_API_BASE}"
    
    # Create test-specific log file
    local log_file="test_logs/${metric_type}_test.log"
    log_info "  Log file: ${log_file}"
    
    # Adjust parameters based on metric type
    local n_metrics_param=$N_METRICS
    if [ "$metric_type" = "finetune" ]; then
        n_metrics_param=1  # Fine-tuning is expensive
        log_info "  Adjusted n_metrics to 1 for finetune type"
    fi
    
    # Count files before test
    local files_before=$(find generated_metrics -name "*.py" -type f | wc -l)
    log_info "  Generated files before test: ${files_before}"
    
    # Run the test
    local start_time=$(date +%s)
    log_info "  Starting metric generation at $(date)"
    
    if python3 metric_generation_demo.py \
        --metric-type "$metric_type" \
        --model qwen \
        --n-metrics "$n_metrics_param" \
        --seed "$TEST_SEED" \
        --test-serialization \
        --model-save-dir "$AUTOMETRICS_MODEL_DIR" \
        --api-base "$QWEN_API_BASE" \
        > "$log_file" 2>&1; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Count files after test
        local files_after=$(find generated_metrics -name "*.py" -type f | wc -l)
        local metric_files_created=$((files_after - files_before))
        
        log_info "  Test completed at $(date)"
        log_info "  Generated files after test: ${files_after}"
        log_info "  Files created during test: ${metric_files_created}"
        
        # Show what files were created (use simpler detection)
        log_info "  Checking for new files in generated_metrics/..."
        if [ $metric_files_created -gt 0 ]; then
            log_info "  New files created:"
            find generated_metrics -name "*.py" -type f -newer test_logs/start_marker 2>/dev/null | head -5 | while read file; do
                log_info "    - $(basename "$file")"
            done
        else
            # Alternative check: just show recent files regardless of timestamp
            log_info "  Recent files in generated_metrics/:"
            ls -t generated_metrics/*.py 2>/dev/null | head -3 | while read file; do
                log_info "    - $(basename "$file")"
            done
        fi
        
        log_success "${description} - PASSED (${duration}s)"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_error "${description} - FAILED after ${duration}s (check ${log_file})"
        log_error "  Test failed at $(date)"
        
        # Show last 20 lines of error log
        log_error "Last 20 lines of error log:"
        tail -n 20 "$log_file" | while IFS= read -r line; do
            echo "    $line"
        done
        
        return 1
    fi
}

run_comprehensive_tests() {
    log_info "Starting comprehensive metric generation tests..."
    log_info "Testing ${#METRIC_TYPES[@]} metric types with serialization..."
    
    # Create a marker file to track newly created files
    touch test_logs/start_marker
    
    local passed_tests=0
    local failed_tests=0
    local total_tests=${#METRIC_TYPES[@]}
    
    # Arrays to track test results
    local passed_list=()
    local failed_list=()
    
    # Test each metric type
    for metric_info in "${METRIC_TYPES[@]}"; do
        local metric_type="${metric_info%:*}"
        local description="${metric_info#*:}"
        
        log_info "=========================================="
        log_info "TEST ${#passed_list[@]}+${#failed_list[@]}+1 of $total_tests"
        log_info "=========================================="
        
        log_info "🚀 About to test: $description ($metric_type)"
        
        # Use || true to prevent early exit on test failure
        if test_metric_type "$metric_type" "$description"; then
            ((passed_tests++))
            passed_list+=("$description")
            log_success "✅ PASSED: $description"
        else
            ((failed_tests++))
            failed_list+=("$description")
            log_error "❌ FAILED: $description"
        fi
        
        # Always show running tally
        log_info "Running tally: ${passed_tests} passed, ${failed_tests} failed"
        log_info "🔄 Continuing to next test..."
        echo ""
    done
    
    log_info "🏁 Completed testing all ${total_tests} metric types!"
    
    # Final comprehensive summary
    log_info "=========================================="
    log_info "FINAL TEST RESULTS"
    log_info "=========================================="
    log_info "Total tests: $total_tests"
    log_success "Passed: $passed_tests"
    if [ $failed_tests -gt 0 ]; then
        log_error "Failed: $failed_tests"
    else
        log_success "Failed: $failed_tests"
    fi
    echo ""
    
    # Detailed breakdown
    if [ ${#passed_list[@]} -gt 0 ]; then
        log_success "PASSED TESTS:"
        for test in "${passed_list[@]}"; do
            log_success "  ✅ $test"
        done
        echo ""
    fi
    
    if [ ${#failed_list[@]} -gt 0 ]; then
        log_error "FAILED TESTS:"
        for test in "${failed_list[@]}"; do
            log_error "  ❌ $test"
        done
        echo ""
        log_error "Check individual log files in test_logs/ for details on failures"
    fi
    
    # Show generated files summary
    local total_generated_files=$(find generated_metrics -name "*.py" -type f | wc -l)
    log_info "Total generated metric files: $total_generated_files"
    if [ $total_generated_files -gt 0 ]; then
        log_info "Recent generated files:"
        find generated_metrics -name "*.py" -type f -newer test_logs/start_marker 2>/dev/null | head -10 | while read file; do
            log_info "  📄 $(basename "$file")"
        done
    fi
    echo ""
    
    # Final result determination
    if [ $failed_tests -eq 0 ]; then
        log_success "🎉 ALL TESTS PASSED! 🎉"
        log_success "All metric generation types are working correctly with serialization"
        return 0
    else
        log_warning "⚠️  SOME TESTS FAILED"
        log_warning "${passed_tests}/${total_tests} tests passed successfully"
        log_warning "The system is partially working but needs attention for failed tests"
        return 1
    fi
}

display_test_info() {
    log_info "=============================================="
    log_info "Comprehensive Metric Generation Test Suite"
    log_info "=============================================="
    log_info ""
    log_info "Configuration:"
    log_info "  - Qwen Model: ${QWEN_MODEL}"
    log_info "  - Qwen Port: ${QWEN_PORT}"
    log_info "  - Prometheus Server: ${PROMETHEUS_SERVER}"
    log_info "  - Test Seed: ${TEST_SEED}"
    log_info "  - Metrics per test: ${N_METRICS}"
    log_info "  - Cache Directory: ${DSPY_CACHEDIR}"
    log_info "  - Model Directory: ${AUTOMETRICS_MODEL_DIR}"
    log_info ""
    log_info "Metric Types to Test:"
    for metric_info in "${METRIC_TYPES[@]}"; do
        local metric_type="${metric_info%:*}"
        local description="${metric_info#*:}"
        log_info "  - ${description} (${metric_type})"
    done
    log_info ""
    log_info "Key Features Being Tested:"
    log_info "  ✓ Metric generation for all types"
    log_info "  ✓ Metric serialization to Python files"
    log_info "  ✓ Metric reloading and execution"
    log_info "  ✓ Metric card generation"
    log_info "  ✓ Reference-based vs reference-free detection"
    log_info "  ✓ Prometheus integration"
    log_info "  ✓ DSPy integration"
    log_info "  ✓ Fine-tuning integration"
    log_info ""
    log_info "=============================================="
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    display_test_info
    
    # Check prerequisites
    check_prerequisites
    
    # Start Qwen server
    start_qwen_server
    
    # Run comprehensive tests - capture exit code but don't exit on failure
    log_info "🎯 Starting comprehensive test suite..."
    if run_comprehensive_tests; then
        # Final success message
        log_success "🎉 Comprehensive metric generation testing completed successfully!"
        log_info "All metric types are rock solid and ready for production use."
        exit 0
    else
        log_warning "⚠️  Some tests failed, but test suite completed"
        log_info "Check individual test logs for details on any failures"
        exit 1
    fi
}

# Execute main function
main "$@" 