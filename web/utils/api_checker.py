"""
API key checker tool
"""

import os

def check_api_keys():
    """Check if all necessary API keys are configured"""

    # Check individual API keys
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    qianfan_key = os.getenv("QIANFAN_API_KEY")

    
    # Build detailed status
    details = {
        "DASHSCOPE_API_KEY": {
            "configured": bool(dashscope_key),
            "display": f"{dashscope_key[:12]}..." if dashscope_key else "Not configured",
            "required": True,
            "description": "Alibaba Dashscope API key"
        },
        "FINNHUB_API_KEY": {
            "configured": bool(finnhub_key),
            "display": f"{finnhub_key[:12]}..." if finnhub_key else "Not configured",
            "required": True,
            "description": "Financial data API key"
        },
        "OPENAI_API_KEY": {
            "configured": bool(openai_key),
            "display": f"{openai_key[:12]}..." if openai_key else "Not configured",
            "required": False,
            "description": "OpenAI API key"
        },
        "ANTHROPIC_API_KEY": {
            "configured": bool(anthropic_key),
            "display": f"{anthropic_key[:12]}..." if anthropic_key else "Not configured",
            "required": False,
            "description": "Anthropic API key"
        },
        "GOOGLE_API_KEY": {
            "configured": bool(google_key),
            "display": f"{google_key[:12]}..." if google_key else "Not configured",
            "required": False,
            "description": "Google AI API key"
        },
        "QIANFAN_ACCESS_KEY": {
            "configured": bool(qianfan_key),
            "display": f"{qianfan_key[:16]}..." if qianfan_key else "Not configured",
            "required": False,
            "description": "Qianfan (ERNIE) API Key (OpenAI compatible), usually starts with bce-v3/"
        },
        # QIANFAN_SECRET_KEY is no longer used for OpenAI compatible path, only kept for script examples
        # "QIANFAN_SECRET_KEY": {
        #     "configured": bool(qianfan_sk),
        #     "display": f"{qianfan_sk[:12]}..." if qianfan_sk else "Not configured",
        #     "required": False,
        #     "description": "Qianfan (ERNIE) Secret Key (for script examples only)"
        # },
    }
    
    # Check required API keys
    required_keys = [key for key, info in details.items() if info["required"]]
    missing_required = [key for key in required_keys if not details[key]["configured"]]
    
    return {
        "all_configured": len(missing_required) == 0,
        "required_configured": len(missing_required) == 0,
        "missing_required": missing_required,
        "details": details,
        "summary": {
            "total": len(details),
            "configured": sum(1 for info in details.values() if info["configured"]),
            "required": len(required_keys),
            "required_configured": len(required_keys) - len(missing_required)
        }
    }

def get_api_key_status_message():
    """Get API key status message"""
    
    status = check_api_keys()
    
    if status["all_configured"]:
        return "✅ All required API keys are configured"
    elif status["required_configured"]:
        return "✅ Required API keys are configured, optional API keys are not configured"
    else:
        missing = ", ".join(status["missing_required"])
        return f"❌ Missing required API keys: {missing}"

def validate_api_key_format(key_type, api_key):
    """Validate API key format"""
    
    if not api_key:
        return False, "API key cannot be empty"
    
    # Basic length check
    if len(api_key) < 10:
        return False, "API key is too short"
    
    # Specific format checks
    if key_type == "DASHSCOPE_API_KEY":
        if not api_key.startswith("sk-"):
            return False, "Alibaba Dashscope API key should start with 'sk-'"
    elif key_type == "OPENAI_API_KEY":
        if not api_key.startswith("sk-"):
            return False, "OpenAI API key should start with 'sk-'"
    elif key_type == "QIANFAN_API_KEY":
        if not api_key.startswith("bce-v3/"):
            return False, "Qianfan API Key (OpenAI compatible) should start with 'bce-v3/'"
    
    return True, "API key format is correct"

def test_api_connection(key_type, api_key):
    """Test API connection (simple validation)"""
    
    # Actual API connection testing can be added here
    # For simplicity, only format validation is done now
    
    is_valid, message = validate_api_key_format(key_type, api_key)
    
    if not is_valid:
        return False, message
    
    # Actual API call testing can be added here
    # For example: call a simple API endpoint to verify key validity
    
    return True, "API key validation passed"
