import os
import sys
import tempfile
import subprocess
import json
from typing import Optional, List, Any
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric

# Try to import DSPy's Python interpreter, but be conservative about when to use it
try:
    from dspy.primitives.python_interpreter import PythonInterpreter as DSPyInterpreter
    DSPY_INTERPRETER_AVAILABLE = True
    # Test if DSPy interpreter can handle our required libraries
    def test_dspy_libraries():
        """Test if DSPy interpreter supports our required libraries"""
        test_code = """
try:
    import numpy
    import math
    import re
    import collections
    result = True
except ImportError:
    result = False
return result
"""
        try:
            with DSPyInterpreter() as interp:
                result = interp.execute(test_code)
                return result
        except:
            return False
    
    DSPY_SUPPORTS_LIBRARIES = test_dspy_libraries()
except ImportError:
    DSPY_INTERPRETER_AVAILABLE = False
    DSPY_SUPPORTS_LIBRARIES = False
    DSPyInterpreter = None

# Always implement our own interpreter as backup
class PythonInterpreter:
    def __init__(self, use_dspy=False):
        self.use_dspy = use_dspy and DSPY_INTERPRETER_AVAILABLE and DSPY_SUPPORTS_LIBRARIES
        if self.use_dspy:
            self._dspy_interpreter = DSPyInterpreter()
        else:
            self._dspy_interpreter = None
    
    def __enter__(self):
        if self.use_dspy and self._dspy_interpreter:
            return self._dspy_interpreter.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_dspy and self._dspy_interpreter:
            return self._dspy_interpreter.__exit__(exc_type, exc_val, exc_tb)
        pass
    
    def execute(self, code, variables=None):
        if self.use_dspy and self._dspy_interpreter:
            # Try DSPy first, fall back to subprocess if it fails
            try:
                return self._dspy_interpreter.execute(code, variables)
            except Exception as e:
                print(f"DSPy interpreter failed, falling back to subprocess: {e}")
                return self._execute_subprocess(code, variables or {})
        else:
            # Use subprocess directly
            return self._execute_subprocess(code, variables or {})
    
    def _execute_subprocess(self, code, variables):
        """Execute code using subprocess for basic sandboxing"""
        # Create a temporary script
        script_template = '''
import sys
import json
import math
import re
import collections
from typing import *

# Import numpy if available (only library we can rely on)
try:
    import numpy
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create dummy numpy for basic operations if not available
    class DummyNumpy:
        def mean(self, arr): return sum(arr) / len(arr) if arr else 0
        def std(self, arr): 
            if not arr: return 0
            mean_val = self.mean(arr)
            return (sum((x - mean_val) ** 2 for x in arr) / len(arr)) ** 0.5
        def array(self, arr): return arr
    numpy = DummyNumpy()
    np = numpy

# Inject variables
{variable_assignments}

# User code
def compute_score_func():
{indented_code}

try:
    result = compute_score_func()
    print(json.dumps({{"result": result, "success": True}}))
except Exception as e:
    print(json.dumps({{"error": str(e), "success": False}}))
'''
        
        # Prepare variable assignments
        var_assignments = []
        for key, value in variables.items():
            if isinstance(value, str):
                var_assignments.append(f'{key} = {repr(value)}')
            elif isinstance(value, list):
                # Handle list variables specially for references
                var_assignments.append(f'{key} = {repr(value)}')
            else:
                var_assignments.append(f'{key} = {value}')
        
        # Indent the user code
        indented_code = '\n'.join('    ' + line for line in code.split('\n'))
        
        script_content = script_template.format(
            variable_assignments='\n'.join(var_assignments),
            indented_code=indented_code
        )
        
        # Write to temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.stdout:
                try:
                    output = json.loads(result.stdout.strip())
                    if output.get("success"):
                        return output["result"]
                    else:
                        raise RuntimeError(f"Code execution failed: {output.get('error', 'Unknown error')}")
                except json.JSONDecodeError:
                    raise RuntimeError(f"Invalid JSON output: {result.stdout}")
            else:
                raise RuntimeError(f"No output from code execution. stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Code execution timed out")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

class GeneratedCodeMetricBase:
    """Base class for generated code metrics with common functionality"""
    
    def __init__(self, name: str, description: str, generated_code: str, task_description: Optional[str] = None, 
                 prefer_dspy_interpreter: bool = False, **kwargs):
        self.generated_code = generated_code
        self.task_description = task_description
        self.prefer_dspy_interpreter = prefer_dspy_interpreter
        self._interpreter = None
        super().__init__(name, description, **kwargs)
        
        # Store code-related parameters for caching
        self._init_params.update({
            'generated_code': generated_code,
            'task_description': task_description,
            'prefer_dspy_interpreter': prefer_dspy_interpreter
        })
    
    def _get_interpreter(self):
        """Get or create a Python interpreter instance"""
        if self._interpreter is None:
            # Default to subprocess interpreter for reliability with all libraries
            # Only use DSPy if explicitly requested and libraries are supported
            use_dspy = self.prefer_dspy_interpreter and DSPY_INTERPRETER_AVAILABLE and DSPY_SUPPORTS_LIBRARIES
            self._interpreter = PythonInterpreter(use_dspy=use_dspy)
        return self._interpreter
    
    def _execute_generated_code(self, input_text: str, output_text: str, references: Optional[List[str]] = None) -> float:
        """Execute the generated code with the given inputs"""
        interpreter = self._get_interpreter()
        
        # Prepare variables for the code execution
        variables = {
            'input': input_text,
            'output': output_text
        }
        
        if references is not None:
            variables['references'] = references
        
        try:
            with interpreter:
                result = interpreter.execute(self.generated_code, variables)
                
                # Ensure result is a number
                if result is None:
                    return 0.0
                elif isinstance(result, (int, float)):
                    return float(result)
                elif isinstance(result, bool):
                    return float(result)
                else:
                    # Try to convert to float
                    try:
                        return float(result)
                    except (ValueError, TypeError):
                        print(f"Warning: Generated code returned non-numeric result: {result}. Using 0.0")
                        return 0.0
                        
        except Exception as e:
            print(f"Error executing generated code for metric {self.name}: {e}")
            return 0.0
    
    def get_generated_code(self) -> str:
        """Get the generated code for inspection"""
        return self.generated_code
    
    def get_task_description(self) -> Optional[str]:
        """Get the task description"""
        return self.task_description
    
    def get_interpreter_info(self) -> dict:
        """Get information about the interpreter being used"""
        return {
            'dspy_available': DSPY_INTERPRETER_AVAILABLE,
            'dspy_supports_libraries': DSPY_SUPPORTS_LIBRARIES if DSPY_INTERPRETER_AVAILABLE else False,
            'prefer_dspy': self.prefer_dspy_interpreter,
            'actual_interpreter': 'dspy' if (self.prefer_dspy_interpreter and DSPY_SUPPORTS_LIBRARIES) else 'subprocess'
        }

class GeneratedCodeReferenceBasedMetric(GeneratedCodeMetricBase, ReferenceBasedMetric):
    """Reference-based metric that executes generated code"""
    
    def __init__(self, name: str, description: str, generated_code: str, task_description: Optional[str] = None, 
                 prefer_dspy_interpreter: bool = False, **kwargs):
        super().__init__(name, description, generated_code, task_description, prefer_dspy_interpreter, **kwargs)
    
    def _calculate_impl(self, input: str, output: str, references: Optional[List[str]] = None, **kwargs) -> float:
        """Calculate the metric using the generated code"""
        return self._execute_generated_code(input, output, references)

class GeneratedCodeReferenceFreeMetric(GeneratedCodeMetricBase, ReferenceFreeMetric):
    """Reference-free metric that executes generated code"""
    
    def __init__(self, name: str, description: str, generated_code: str, task_description: Optional[str] = None, 
                 prefer_dspy_interpreter: bool = False, **kwargs):
        super().__init__(name, description, generated_code, task_description, prefer_dspy_interpreter, **kwargs)
    
    def _calculate_impl(self, input: str, output: str, references: Optional[List[str]] = None, **kwargs) -> float:
        """Calculate the metric using the generated code"""
        # For reference-free metrics, we don't pass references
        return self._execute_generated_code(input, output, None) 