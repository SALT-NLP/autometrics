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

def safe_model_to_device(model: torch.nn.Module, device: Union[str, torch.device], 
                        model_name: str = "model") -> torch.nn.Module:
    """
    Safely move a model to a device, handling meta tensor issues.
    
    This function attempts to move a model to the specified device using the standard
    .to() method, but falls back to .to_empty() if a meta tensor error occurs.
    
    Args:
        model: The PyTorch model to move
        device: The target device (string or torch.device)
        model_name: Name of the model for error messages
        
    Returns:
        The model moved to the target device
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    try:
        # Try standard .to() method first
        return model.to(device)
    except NotImplementedError as e:
        # Handle meta tensor issue
        if "Cannot copy out of meta tensor" in str(e):
            print(f"    🔧 Meta tensor issue detected for {model_name}, using to_empty()...")
            # Use to_empty() to properly move from meta to device
            return model.to_empty(device=device)
        else:
            # Re-raise if it's a different NotImplementedError
            raise e
    except Exception as e:
        # Handle any other device-related errors
        print(f"    ⚠ Device mapping issue for {model_name}: {e}")
        print(f"    🔧 This may be due to device mapping conflicts in parallel execution")
        raise e

def safe_model_loading(model_class, model_name_or_path: str, device: Union[str, torch.device] = None,
                      **kwargs) -> torch.nn.Module:
    """
    Safely load a model with proper device handling.
    
    This function loads a model and safely moves it to the specified device,
    handling meta tensor issues that commonly occur with device_map parameters.
    
    Args:
        model_class: The model class to instantiate (e.g., AutoModelForSequenceClassification)
        model_name_or_path: The model name or path to load
        device: The target device (if None, uses CUDA if available, otherwise CPU)
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        The loaded model on the target device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    try:
        # First attempt: try loading with provided kwargs
        model = model_class.from_pretrained(model_name_or_path, **kwargs)
        model.eval()
        
        # Check if model needs to be moved to device
        if not hasattr(model, 'hf_device_map') or model.hf_device_map is None:
            # Model is not device-mapped, move it to the target device
            model = safe_model_to_device(model, device, model_name_or_path)
        
        return model
        
    except NotImplementedError as e:
        # Handle meta tensor issue
        if "Cannot copy out of meta tensor" in str(e):
            print(f"    🔧 Meta tensor issue detected for {model_name_or_path}, using to_empty()...")
            
            # Remove device_map from kwargs if present to avoid conflicts
            load_kwargs = kwargs.copy()
            if 'device_map' in load_kwargs:
                del load_kwargs['device_map']
            
            # Load model without device_map first
            model = model_class.from_pretrained(model_name_or_path, **load_kwargs)
            model.eval()
            
            # Use to_empty() to properly move from meta to device
            model = model.to_empty(device=device)
            
            return model
        else:
            raise e
    except Exception as e:
        print(f"    ❌ Failed to load {model_name_or_path}: {e}")
        raise e 