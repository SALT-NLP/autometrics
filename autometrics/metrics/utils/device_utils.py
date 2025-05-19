import torch
from typing import Union, Optional, Any, Dict

def get_model_device(model: Any, fallback_device: Optional[torch.device] = None) -> torch.device:
    """
    Determine the device a model is on by checking various attributes.
    
    Args:
        model: The model to check
        fallback_device: Fallback device if no device can be determined from the model
        
    Returns:
        torch.device: The device the model is on
    """
    # Check direct device property
    if hasattr(model, "device") and model.device is not None:
        return model.device
    
    # Check for get_device method and make sure it's callable
    if hasattr(model, "get_device") and callable(model.get_device):
        try:
            device = model.get_device()
            if device is not None:
                return device
        except Exception:
            pass  # Fall through to next method
    
    # Try to find a parameter's device
    try:
        for param in model.parameters():
            if hasattr(param, "device"):
                return param.device
    except (StopIteration, AttributeError, TypeError):
        pass  # Fall through to fallback
    
    # Fall back to provided device or CPU
    if fallback_device is not None:
        return fallback_device
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_tensor_on_device(tensor: Union[torch.Tensor, Dict, Any], 
                            device: torch.device) -> Union[torch.Tensor, Dict, Any]:
    """
    Ensure that a tensor or dictionary of tensors is on the specified device.
    If the input is not a tensor or dictionary, it is returned unchanged.
    
    Args:
        tensor: The tensor, dictionary of tensors, or other object
        device: The target device
        
    Returns:
        The tensor or dict of tensors on the specified device, or the unchanged input
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, dict):
        # Handle dictionaries of tensors (common in HuggingFace models)
        return {k: ensure_tensor_on_device(v, device) for k, v in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        # Handle lists or tuples of tensors
        tensor_type = type(tensor)
        return tensor_type([ensure_tensor_on_device(t, device) for t in tensor])
    return tensor 