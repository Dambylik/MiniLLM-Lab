# Example: testing validate_and_coerce interactively

from src.validation import build_functions_lookup, validate_and_coerce

functions = [
    {"fn_name": "fn_add_numbers", "args_types": {"a": "float", "b": "float"}, "return_type": "float"},
    {"fn_name": "fn_reverse_string", "args_types": {"s": "str"}, "return_type": "str"},
    {"fn_name": "fn_greet", "args_types": {"name": "str"}, "return_type": "str"}
]

lookup = build_functions_lookup(functions)

# example candidate where numbers are strings — should be coerced
candidate = {
    "prompt": "Add 2 and 3",
    "fn_name": "fn_add_numbers",
    "args": {"a": "2", "b": 3}
}

ok, result_or_errors = validate_and_coerce(candidate, lookup)
if ok:
    print("Validated and coerced result:", result_or_errors)
else:
    print("Validation failed:", result_or_errors)
