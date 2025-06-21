from datetime import datetime
from typing import Optional
import dspy

__all__ = ["build_llm_judge_metric_card"]

def generate_llm_constructor_code(model: dspy.LM) -> str:
    model_name = str(getattr(model, "model", model))
    kwargs = {}
    
    # Extract all kwargs if present
    if hasattr(model, "kwargs"):
        kwargs = {k: v for k, v in model.kwargs.items() if v is not None}
    
    """Generate constructor code for DSPy LLMs"""
    if "openai" in model_name.lower():
        kwargs["api_key"] = "os.getenv(\"OPENAI_API_KEY\")"
    elif "anthropic" in model_name.lower():
        kwargs["api_key"] = "os.getenv(\"ANTHROPIC_API_KEY\")"
    elif "gemini" in model_name.lower():
        kwargs["api_key"] = "os.getenv(\"GEMINI_API_KEY\")"
    else:
        kwargs["api_key"] = "None"
        
    kwargs_str = ", ".join(f"{k}={v}" if type(v) != str else f"{k}='{v}'" for k, v in kwargs.items())
    return f"dspy.LM(model=\'{model_name}\', {kwargs_str})"