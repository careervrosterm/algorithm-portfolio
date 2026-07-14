import json

output_schema = {
    "category": {"Communications", "Operations", "Finances", "Tech Issues", "Customer", "Data", "Documents", "Other"},
    "priority": {"Low", "Medium", "High", "Unknown"},
    "risk": {"Low", "Medium", "High", "Unknown"},
    "suggested_automation": str,
    "reasoning": str,
    "needs_human_review": bool
}

acceptable_priority = output_schema["priority"]
acceptable_risk = output_schema["risk"]
acceptable_category = output_schema["category"]

def validate(response):
    try:
        response = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e}")
    if not isinstance(response, dict):
        raise TypeError("JSON must be converted into a dictionary.")
    
    missing_keys = output_schema.keys() - response.keys()
    extra_keys = response.keys() - output_schema.keys()

    if missing_keys:
        raise ValueError(f"Missing fields: {missing_keys}")
    if extra_keys:
        raise ValueError(f"Unexpected fields: {extra_keys}")
    for field, rule in output_schema.items():
        value = response[field]

        if not (isinstance(rule,set) or isinstance(rule,type)):
            raise TypeError(f"Given schema rule {rule} is neither a set or python type.")
        if isinstance(rule, set) and value not in rule:
            raise ValueError(f"Invalid value for {field}: {value}")

        if isinstance(rule, type) and not isinstance(value, rule):
            raise TypeError(
                f"Invalid type for {field}: expected {rule.__name__}, got {type(value).__name__}"
            )
        
    if response["priority"] == "Unknown" and response["risk"] == "Unknown":
        response["needs_human_review"] = True

    return response