import json
import re
import time
from typing import Optional, List, Any
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric

# Import our custom interpreter first
try:
    from autometrics.util.custom_python_interpreter import CustomPythonInterpreter
    CUSTOM_INTERPRETER_AVAILABLE = True
except ImportError:
    CUSTOM_INTERPRETER_AVAILABLE = False
    CustomPythonInterpreter = None

# Import DSPy's Python interpreter as fallback
try:
    from dspy.primitives.python_interpreter import PythonInterpreter as DSPyInterpreter
    DSPY_INTERPRETER_AVAILABLE = True
except ImportError:
    DSPY_INTERPRETER_AVAILABLE = False
    DSPyInterpreter = None

class SecurityError(Exception):
    """Raised when secure execution is not available"""
    pass

class GeneratedCodeMetricBase:
    """Base class for generated code metrics with common functionality"""
    
    def __init__(self, name: str, description: str, generated_code: str, task_description: Optional[str] = None, **kwargs):
        self.generated_code = generated_code
        self.task_description = task_description
        self._interpreter = None
        super().__init__(name, description, **kwargs)
        
        # Store code-related parameters for caching
        self._init_params.update({
            'generated_code': generated_code,
            'task_description': task_description
        })
    
    def _get_interpreter(self):
        """Get or create an interpreter instance, preferring our custom one"""
        if self._interpreter is None:
            # Prefer our custom interpreter with enhanced package loading
            if CUSTOM_INTERPRETER_AVAILABLE:
                self._interpreter = CustomPythonInterpreter()
            elif DSPY_INTERPRETER_AVAILABLE:
                self._interpreter = DSPyInterpreter()
            else:
                raise SecurityError(
                    "No Python interpreter available. Please install DSPy for secure code execution: "
                    "pip install dspy-ai"
                )
        return self._interpreter
    
    def _parse_generated_code(self, code: str) -> tuple[str, str]:
        """
        Parse generated code to separate imports from the main logic.
        Returns (imports_section, logic_section)
        """
        lines = code.strip().split('\n')
        import_lines = []
        logic_lines = []
        
        # Track if we're still in the imports section
        in_imports = True
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if this line is an import statement
            if (stripped_line.startswith('import ') or 
                stripped_line.startswith('from ') or
                stripped_line == '' and in_imports):  # Empty lines in imports section
                import_lines.append(line)
            else:
                # Once we hit non-import code, everything else is logic
                in_imports = False
                logic_lines.append(line)
        
        imports_section = '\n'.join(import_lines)
        logic_section = '\n'.join(logic_lines)
        
        return imports_section.strip(), logic_section.strip()
    
    def _create_function_signature(self, has_references: bool) -> str:
        """Create the appropriate function signature based on metric type"""
        if has_references:
            return "def compute_score(input, output, references=None):"
        else:
            return "def compute_score(input, output):"
    
    def _execute_generated_code(self, input_text: str, output_text: str, references: Optional[List[str]] = None) -> float:
        """Execute the generated code with the given inputs"""
        # Use the batched version with a single item to avoid code duplication
        inputs = [input_text]
        outputs = [output_text]
        references_list = [references] if references is not None else None
        
        results = self._execute_generated_code_batched(inputs, outputs, references_list)
        return results[0] if results else 0.0

    def _execute_generated_code_batched(self, inputs: List[str], outputs: List[str], references_list: Optional[List[List[str]]] = None) -> List[float]:
        """Execute the generated code with batched inputs for better performance"""
        
        if not inputs or len(inputs) != len(outputs):
            raise ValueError("inputs and outputs must be non-empty lists of the same length")
        
        if references_list is not None and len(references_list) != len(inputs):
            raise ValueError("references_list must be None or the same length as inputs")
        
        interpreter = self._get_interpreter()
        
        # Parse the generated code to separate imports from logic (setup once)
        imports_section, logic_section = self._parse_generated_code(self.generated_code)
        
        has_references = references_list is not None
        
        # Create the function signature (setup once)
        function_signature = self._create_function_signature(has_references)
        
        # Indent the logic section for function body (setup once)
        indented_logic = self._indent_code(logic_section) if logic_section else "    return 0.0"
        
        # Build the function definition part (setup once)
        setup_code_parts = []
        
        # Add imports at module level (unindented)
        if imports_section:
            setup_code_parts.append(imports_section)
            setup_code_parts.append("")  # Empty line after imports
        
        # Add the function definition
        setup_code_parts.append(function_signature)
        setup_code_parts.append(indented_logic)
        setup_code_parts.append("")  # Empty line after function
        
        setup_code = '\n'.join(setup_code_parts)
        
        # Lint the setup code before execution
        lint_result = self._lint_code(setup_code + "\nresult = 0.0\nresult")
        if not lint_result['valid']:
            raise SyntaxError(f"Generated code has syntax errors: {lint_result['error']}")
        
        # Execute the setup code once to define the function and load packages
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Execute setup code to define the function
                setup_result = interpreter.execute(setup_code, {})
                
                # Check if this still looks like a loading message (edge case)
                if isinstance(setup_result, str) and self._is_loading_message(setup_result):
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                
                break  # Setup successful
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise e
        
        # Now execute the function for each input/output pair
        results = []
        
        for i in range(len(inputs)):
            input_text = inputs[i]
            output_text = outputs[i]
            references = references_list[i] if references_list else None
            
            # Prepare variables for this execution
            variables = {
                'input': input_text,
                'output': output_text
            }
            
            if has_references:
                variables['references'] = references
            
            # Create the function call code
            if has_references:
                call_code = "result = compute_score(input, output, references)\nresult"
            else:
                call_code = "result = compute_score(input, output)\nresult"
            
            # Execute the function call
            try:
                result = interpreter.execute(call_code, variables)
                
                # Convert to numeric result
                numeric_result = self._ensure_numeric_result(result)
                results.append(numeric_result)
                
            except Exception as e:
                print(f"Error executing batched code for item {i}: {e}")
                results.append(0.0)
        
        return results
    
    def _is_loading_message(self, result) -> bool:
        """Check if a result looks like a package loading message"""
        if not isinstance(result, str):
            return False
        
        result_lower = result.lower()
        loading_indicators = [
            "loading ",
            "downloading ",
            "installing ",
            "cdn.jsdelivr.net",
            ".whl",
            "pyodide",
            "fetching",
            "regex",
            "sqlite3",
            "nltk",
            "package",
            "wheel",
            "download",
            "failed to load"
        ]
        return any(indicator in result_lower for indicator in loading_indicators)
    
    def _ensure_numeric_result(self, result) -> float:
        """Ensure the result is a numeric value"""
        if result is None:
            return 0.0
        elif isinstance(result, (int, float)):
            return float(result)
        elif isinstance(result, bool):
            return float(result)
        else:
            try:
                return float(result)
            except (ValueError, TypeError):
                print(f"Generated code returned non-numeric result: {result}")
                return 0.0
    
    def _indent_code(self, code: str) -> str:
        """Indent code for function wrapping - adds 4 spaces to ALL lines"""
        if not code.strip():
            return "    return 0.0"
        
        lines = code.split('\n')
        indented_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty line - always add 4 spaces
                indented_lines.append('    ' + line)
            else:  # Empty line
                indented_lines.append('')
        
        return '\n'.join(indented_lines)
    
    def get_generated_code(self) -> str:
        """Get the generated code for inspection"""
        return self.generated_code
    
    def get_task_description(self) -> Optional[str]:
        """Get the task description"""
        return self.task_description
    
    def get_interpreter_info(self) -> dict:
        """Get information about the interpreter being used"""
        return {
            'custom_available': CUSTOM_INTERPRETER_AVAILABLE,
            'dspy_available': DSPY_INTERPRETER_AVAILABLE,
            'interpreter_type': 'custom' if CUSTOM_INTERPRETER_AVAILABLE else ('dspy' if DSPY_INTERPRETER_AVAILABLE else 'none')
        }
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports
    
    def _preload_packages(self, input_text: str, output_text: str, references: Optional[List[str]] = None):
        """Pre-load packages using proper Pyodide package loading sequence"""
        imports = self._extract_imports(self.generated_code)
        if not imports:
            return
            
        # Extract package names that need to be installed
        packages_to_install = set()
        for import_line in imports:
            if import_line.startswith('import '):
                # Extract package name from "import package" or "import package.module"
                package = import_line[7:].split('.')[0].split(' as ')[0].strip()
                if package not in ['sys', 'os', 'math', 're', 'json', 'time', 'collections', 'itertools']:
                    # Skip built-in packages
                    packages_to_install.add(package)
            elif import_line.startswith('from '):
                # Extract package name from "from package import ..."
                package = import_line[5:].split('.')[0].split(' ')[0].strip()
                if package not in ['sys', 'os', 'math', 're', 'json', 'time', 'collections', 'itertools']:
                    packages_to_install.add(package)
        
        if not packages_to_install:
            return
        
        interpreter = self._get_interpreter()
        
        variables = {
            'input': input_text,
            'output': output_text
        }
        if references is not None:
            variables['references'] = references
            
        # Use the correct Pyodide package loading sequence
        for package in packages_to_install:
            # First try loadPackage() for built-in Pyodide packages
            load_code = f"""
import pyodide_js
await pyodide_js.loadPackage("{package}")
print(f"Successfully loaded {package} via loadPackage")
"""
            try:
                result = interpreter.execute(load_code, variables)
                if not self._is_loading_message(result):
                    continue  # Success, move to next package
            except Exception as e:
                pass
            
            # If loadPackage fails, try micropip.install()
            install_code = f"""
import pyodide_js
await pyodide_js.loadPackage("micropip")
import micropip
await micropip.install("{package}")
print(f"Successfully installed {package} via micropip")
"""
            try:
                result = interpreter.execute(install_code, variables)
            except Exception as e:
                continue

    def _lint_code(self, code: str) -> dict:
        """Lint the code using Python's ast module to catch syntax errors"""
        import ast
        import re
        
        try:
            # First, check for obvious indentation issues
            lines = code.split('\n')
            in_function = False
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Skip empty lines and comments
                if not stripped or stripped.startswith('#'):
                    continue
                
                # Check for mixed tabs and spaces (common issue)
                if '\t' in line and '    ' in line:
                    return {
                        'valid': False,
                        'error': f"Line {i}: Mixed tabs and spaces detected"
                    }
                
                # Track if we're inside a function
                if stripped.startswith('def '):
                    in_function = True
                    continue
                elif not line.startswith((' ', '\t')) and not stripped.startswith(('import ', 'from ')):
                    # This is a top-level statement, we're no longer in a function
                    in_function = False
                
                # Check for indentation issues inside functions
                if in_function and stripped:
                    # Inside a function, code should be indented
                    if not line.startswith(('    ', '\t')):
                        return {
                            'valid': False,
                            'error': f"Line {i}: Code inside function should be indented - '{stripped[:50]}'"
                        }
            
            # Check for common variable reference issues
            # Look for patterns like using WordNetLemmatizer without instantiation
            common_class_patterns = [
                (r'lemmatizer\.(lemmatize|pos_tag)', 'lemmatizer = WordNetLemmatizer()'),
                (r'vectorizer\.(fit_transform|transform)', 'vectorizer = TfidfVectorizer()'),
                (r'stemmer\.(stem)', 'stemmer = PorterStemmer()'),
            ]
            
            for pattern, suggestion in common_class_patterns:
                if re.search(pattern, code):
                    # Check if the variable is defined
                    var_name = pattern.split('.')[0].replace('\\', '')
                    if f'{var_name} =' not in code:
                        return {
                            'valid': False,
                            'error': f"Variable '{var_name}' is used but not defined. Add: {suggestion}"
                        }
            
            # Try to parse with ast
            ast.parse(code)
            
            return {'valid': True, 'error': None}
            
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f" (in: '{e.text.strip()}')"
            return {
                'valid': False,
                'error': error_msg
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Linting error: {str(e)}"
            }

class GeneratedCodeReferenceBasedMetric(GeneratedCodeMetricBase, ReferenceBasedMetric):
    """Reference-based metric that executes generated code"""
    
    def __init__(self, name: str, description: str, generated_code: str, task_description: Optional[str] = None, **kwargs):
        super().__init__(name, description, generated_code, task_description, **kwargs)
    
    def _calculate_impl(self, input: str, output: str, references: Optional[List[str]] = None, **kwargs) -> float:
        """Calculate the metric using the generated code"""
        return self._execute_generated_code(input, output, references)

    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references_list: Optional[List[List[str]]] = None, **kwargs) -> List[float]:
        """Calculate the metric using the generated code for multiple inputs efficiently"""
        return self._execute_generated_code_batched(inputs, outputs, references_list)

class GeneratedCodeReferenceFreeMetric(GeneratedCodeMetricBase, ReferenceFreeMetric):
    """Reference-free metric that executes generated code"""
    
    def __init__(self, name: str, description: str, generated_code: str, task_description: Optional[str] = None, **kwargs):
        super().__init__(name, description, generated_code, task_description, **kwargs)
    
    def _calculate_impl(self, input: str, output: str, references: Optional[List[str]] = None, **kwargs) -> float:
        """Calculate the metric using the generated code"""
        # For reference-free metrics, we don't pass references
        return self._execute_generated_code(input, output, None)

    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references_list: Optional[List[List[str]]] = None, **kwargs) -> List[float]:
        """Calculate the metric using the generated code for multiple inputs efficiently"""
        # For reference-free metrics, we don't pass references
        return self._execute_generated_code_batched(inputs, outputs, None) 