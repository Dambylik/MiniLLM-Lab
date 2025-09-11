# (entry point)
# Reads input files.
# Sends prompts to the LLM.
# Gets candidate JSON output.
# Calls functions in validation.py to check that output.
# If valid → saves to output/
# If invalid → logs error and retries.

from llm_sdk import Small_LLM_Model
from .tokenizer_stub import StubTokenizer
import json
import sys
import numpy as np
from pathlib import Path
from .validation import build_functions_lookup, validate_and_coerce
from .utils import safe_parse_json


def generate_one_token(prompt: str, model, tokenizer):
    input_ids = tokenizer.encode(prompt)
    logits = model.get_logits_from_input_ids(input_ids)
    next_token_id = int(np.argmax(logits))
    new_ids = input_ids + [next_token_id]
    decoded_text = tokenizer.decode(new_ids)
    return decoded_text, new_ids



def load_json_file(path: Path):
    """Load a JSON file safely, with error handling."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}", file=sys.stderr)
        


def main():
    # Step 1: locate input files
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    tests_file = input_dir / "function_calling_tests.json"
    defs_file = input_dir / "functions_definition.json"
    output_file = output_dir / "function_calling_name.json"
    
    function_defs = load_json_file(defs_file)
    functions_lookup = build_functions_lookup(function_defs)
    
    test_prompts = load_json_file(tests_file)
    results = []
    model = Small_LLM_Model()
    tokenizer = StubTokenizer()
    
    # print("Loaded test prompts:", test_prompts)
    # print("Loaded function definitions:", function_defs)
    
    for prompt_dict in test_prompts:
        prompt = prompt_dict["prompt"]
        decoded, ids = generate_one_token(prompt, model, tokenizer)
        raw_output = """{
            "prompt": "%s",
            "fn_name": "fn_add_numbers",
            "args": {"a": 2, "b": 3}
        }""" % prompt

        print(f"Prompt: {prompt}")
        print(f"Decoded: {decoded}")

        parsed = safe_parse_json(raw_output)
        if parsed is None:
            print(f"Warning: could not extract JSON for prompt: {prompt}")
            continue
        
        ok, result_or_errors = validate_and_coerce(parsed, functions_lookup)
        if ok:
            results.append(result_or_errors)
        else:
            print(f"Validation failed for prompt {prompt}: {result_or_errors}")
     
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_file}")

    
    
    
if __name__ == "__main__":
    main()