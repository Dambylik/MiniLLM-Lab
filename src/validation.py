# Implements the step-by-step checks
# Uses the models from models.py.

from typing import Any, Dict, List, Tuple
from pydantic import ValidationError
from .models import FunctionCall, FunctionDefinition


def build_functions_lookup(defs: List[Dict[str, Any]]) -> Dict[str, FunctionDefinition]:
    """
    Convert a list of dicts (loaded from functions_definition.json)
    into a lookup: fn_name -> FunctionDefinition (validated).
    """
    lookup: Dict[str, FunctionDefinition] = {}
    for d in defs:
        # Validate the structure of each function definition
        fd = FunctionDefinition(**d)
        lookup[fd.fn_name] = fd
    return lookup


def coerce_value(value: Any, expected_type: str) -> Tuple[Any, List[str]]:
    """
    Try to coerce `value` into `expected_type`.
    Returns (coerced_value, errors_list). If coercion failed, coerced_value is None.
    Supported expected_type strings: "float", "int", "str", "bool".
    """
    errors: List[str] = []
    t = expected_type.strip().lower()
    
    try:
        if t in ("float", "double"):
            if isinstance(value, (int, float)):
                return float(value), []
            if isinstance(value, str):
                return float(value.strip()), []
            raise ValueError("cannot convert to float")
        
        if t == "int":
            if isinstance(value, int):
                return int(value), []
            if isinstance(value, float):
                if value.is_integer():
                    return int(value), []
                raise ValueError("float not integral for int target")
            if isinstance(value, str):
                # allow "3" or "3.0" that is integral
                fv = float(value.strip())
                if fv.is_integer():
                    return int(fv), []
                raise ValueError("string float not integral for int target")
            raise ValueError("cannot convert to int")
        
        if t in ("str", "string"):
            # permissive: convert other types to string
            if isinstance(value, str):
                return value, []
            return str(value), []
        
        if t == "bool":
            if isinstance(value, bool):
                return value, []
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("true", "1"):
                    return True, []
                if s in ("false", "0"):
                    return False, []
                raise ValueError("cannot parse boolean from string")
            if isinstance(value, int):
                if value in (0, 1):
                    return bool(value), []
                raise ValueError("cannot convert to bool")    
        
        raise ValueError(f"unsupported expected type '{expected_type}'")
    
    except Exception as exc:
        errors.append(str(exc))
        return None, errors


def validate_and_coerce(
    candidate_dict: Dict[str, Any],
    functions_lookup: Dict[str, FunctionDefinition],
) -> Tuple[bool, Any]:
    """
    Validate a candidate dict from the LLM and coerce its args to the expected types.
    Returns (True, coerced_result_dict) on success, or (False, list_of_errors) on failure.
    coerced_result_dict has the exact schema:
      { "prompt": str, "fn_name": str, "args": { ... coerced values ... } }
    """
    errors: List[str] = []
    
    # Step 1: candidate must be a dict and have the exact top-level keys
    if not isinstance(candidate_dict, dict):
        return False, ["Output is not a Json object"]
    
    required_keys = {"prompt", "fn_name", "args"}
    keys = set(candidate_dict.keys())
    if keys != required_keys:
        missing = required_keys - keys
        extra = keys - required_keys
        if missing:
            errors.append("Missing top-level keys: " + ", ".join(sorted(missing)))
        if extra:
            errors.append("Extra top-level keys: " + ", ".join(sorted(extra)))
        return False, errors
    
    # Step 2: structural validation (Pydantic ensures prompt=str, fn_name=str, args=dict)
    try:
        fc = FunctionCall(**candidate_dict)
    except ValidationError as exc:
        return False, [str(exc)]
    
    # Step 3: fn_name must exist in the catalog
    if fc.fn_name not in functions_lookup:
        return False, [f"fn_name '{fc.fn_name}' not found in function definitions"]
    
    expected_types = functions_lookup[fc.fn_name].args_types
    
    # Step 4: arg names must match exactly
    arg_keys = set(fc.args.keys())
    expected_keys = set(expected_types.keys())
    if arg_keys != expected_keys:
        missing = expected_keys - arg_keys
        extra = arg_keys - expected_keys
        if missing:
            errors.append("Missing arguments: " + ", ".join(sorted(missing)))
        if extra:
            errors.append("Extra arguments: " + ", ".join(sorted(extra)))
        return False, errors
    
    # Step 5: coerce each arg to the expected type
    coerced_args: Dict[str, Any] = {}
    for arg_name, expected_type in expected_types.items():
        raw_value = fc.args[arg_name]
        coerced, err = coerce_value(raw_value, expected_type)
        if err:
            errors.append(f"Argument '{arg_name}': " + "; ".join(err))
        else:
            coerced_args[arg_name] = coerced

    if errors:
        return False, errors
    
    # All is checked — build the canonical result
    result = {"prompt": fc.prompt, "fn_name": fc.fn_name, "args": coerced_args}
    return True, result