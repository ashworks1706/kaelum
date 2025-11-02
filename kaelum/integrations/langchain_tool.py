"""LangChain integration for KaelumAI"""

from typing import Optional, Type
from pydantic import BaseModel, Field
from kaelum import kaelum_enhance_reasoning
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun

class KaelumReasoningInput(BaseModel):
    """Input schema for Kaelum reasoning tool."""
    query: str = Field(
        description="The question or problem requiring step-by-step reasoning. "
        "Best for: math problems, logical puzzles, multi-step analysis, complex decision-making."
    )
    domain: Optional[str] = Field(
        default="general",
        description="Domain context for verification: 'math' (enables symbolic verification), "
        "'logic', 'code', 'science', or 'general'. Use 'math' for equations and calculations."
    )


class KaelumReasoningTool(BaseTool):
    
    name: str = "kaelum_reasoning"
    description: str = (
        "Calls a local reasoning engine (KaelumAI) to generate verified step-by-step reasoning traces. "
        "Use this tool when facing complex problems that require careful logical breakdown: "
        "mathematical calculations, multi-step word problems, logical puzzles, algorithmic thinking, "
        "or any task where step-by-step verification improves accuracy. "
        "\n\n"
        "The tool returns: (1) verified reasoning steps from a local LLM, (2) symbolic verification results "
        "for math operations, and (3) a suggested approach. You should then use these verified steps to "
        "construct your final answer. "
        "\n\n"
        "When to use: Problems requiring >2 logical steps, math with equations, situations where accuracy "
        "is critical and you need verified intermediate steps."
    )
    args_schema: Type[BaseModel] = KaelumReasoningInput
    return_direct: bool = False
    
    def _run(
        self,
        query: str,
        domain: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = kaelum_enhance_reasoning(query=query, domain=domain)
        
        # Format output for commercial LLM consumption
        output = "═══ KAELUM VERIFIED REASONING ═══\n\n"
        output += f"Query: {query}\n"
        output += f"Domain: {domain}\n\n"
        
        output += "Step-by-Step Reasoning (Verified):\n"
        for i, step in enumerate(result["reasoning_steps"], 1):
            output += f"  {i}. {step}\n"
        
        output += f"\n💡 Suggested Final Approach:\n{result['suggested_approach']}\n"
        output += f"\n📊 Reasoning Depth: {result['reasoning_count']} steps\n"
        output += "\nℹ️  Note: These steps have been verified by Kaelum's symbolic and consistency checkers.\n"
        output += "Use them to construct an accurate, well-reasoned final answer."
        
        return output
    
    async def _arun(
        self,
        query: str,
        domain: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async version (calls sync for now)."""
        return self._run(query, domain, run_manager)


# Simple function-based tool creator (alternative to class)
def create_kaelum_tool():
    try:
        from langchain.tools import Tool
    except ImportError:
        raise ImportError("LangChain not installed. Install with: pip install langchain")
    
    def kaelum_func(query: str) -> str:
        """Wrapper function for Kaelum reasoning."""
        result = kaelum_enhance_reasoning(query=query, domain="general")
        
        output = "═══ KAELUM VERIFIED REASONING ═══\n\n"
        output += "Reasoning Steps:\n"
        for i, step in enumerate(result["reasoning_steps"], 1):
            output += f"{i}. {step}\n"
        output += f"\nSuggested Approach: {result['suggested_approach']}\n"
        output += f"Steps: {result['reasoning_count']}"
        
        return output
    
    return Tool(
        name="KaelumReasoning",
        func=kaelum_func,
        description=(
            "Generates verified step-by-step reasoning using a local LLM with symbolic verification. "
            "Use for math problems, logical puzzles, multi-step analysis where accuracy matters. "
            "Input should be the question needing detailed reasoning. "
            "Returns verified reasoning steps and a suggested approach."
        )
    )
