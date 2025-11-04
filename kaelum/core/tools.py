"""Function calling tools for commercial LLMs to use Kaelum reasoning."""

from typing import Dict, List, Any


def get_kaelum_function_schema() -> Dict[str, Any]:
    """
    Get the function schema for Kaelum reasoning enhancement.
    This can be passed to commercial LLMs (Gemini, GPT-4, Claude, etc.) 
    as a function/tool they can call.
    
    Returns:
        Function schema compatible with OpenAI/Gemini function calling format
    """
    return {
        "name": "kaelum_enhance_reasoning",
        "description": (
            "Enhances reasoning for complex questions by breaking them down into "
            "logical steps. Use this when you need to solve math problems, "
            "logical puzzles, multi-step reasoning tasks, or any question that "
            "requires careful step-by-step thinking. Returns a structured reasoning "
            "trace that you can use to formulate your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or problem that needs reasoning enhancement"
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain hint (e.g., 'math', 'logic', 'code', 'science')",
                    "enum": ["math", "logic", "code", "science", "general"]
                }
            },
            "required": ["query", "domain"]
        }
    }
