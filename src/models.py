# Defines Pydantic models (FunctionDefinition, FunctionCall).
# Both main.py and validation.py will import from here.

from typing import Any, Dict
from pydantic import BaseModel

class FunctionDefinition(BaseModel):
    """
    Represents one entry from functions_definition.json
    - fn_name: exact function identifier
    - args_types: mapping arg_name -> type name (e.g. "a": "float")
    - return_type: string describing return type
    """
    fn_name: str
    args_types: Dict[str, str]
    return_type: str


class FunctionCall(BaseModel):
    """
    Represents a candidate function call produced by the LLM:
    Must contain exactly the keys: prompt, fn_name, args
    """
    prompt: str
    fn_name: str
    args: Dict[str, Any]

    
class Config:
    # forbid extra keys so stray fields are rejected early
    extra = "forbid"
