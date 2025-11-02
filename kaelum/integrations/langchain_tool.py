"""LangChain integration for KaelumAI"""

from typing import Optional, Type
from pydantic import BaseModel, Field
from kaelum import kaelum_enhance_reasoning
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

LANGCHAIN_AVAILABLE = True

class KaelumReasoningInput(BaseModel):
    """Input schema for Kaelum reasoning tool."""
    query: str = Field(description="The question or problem that needs reasoning enhancement")
    domain: Optional[str] = Field(
        default="general",
        description="Optional domain hint: math, logic, code, science, or general"
    )


class KaelumReasoningTool(BaseTool):
    """
    LangChain tool for Kaelum reasoning enhancement.
    
    Usage:
        from kaelum.integrations.langchain_tool import KaelumReasoningTool
        from kaelum import set_reasoning_model
        
        # Configure Kaelum
        set_reasoning_model(provider="ollama", model="qwen2.5:7b")
        
        # Create tool
        kaelum_tool = KaelumReasoningTool()
        
        # Use in agent
        tools = [kaelum_tool]
        agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION)
    """
    
    name: str = "kaelum_reasoning"
    description: str = (
        "Enhances reasoning for complex questions by breaking them down into "
        "logical steps. Use this when you need to solve math problems, "
        "logical puzzles, multi-step reasoning tasks, or any question that "
        "requires careful step-by-step thinking. Returns structured reasoning "
        "steps and a suggested approach."
    )
    args_schema: Type[BaseModel] = KaelumReasoningInput
    return_direct: bool = False
    
    def _run(
        self,
        query: str,
        domain: str = "general",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute Kaelum reasoning."""
        result = kaelum_enhance_reasoning(query=query, domain=domain)
        
        # Format output for LangChain
        output = "🧠 Kaelum Reasoning Steps:\n\n"
        for i, step in enumerate(result["reasoning_steps"], 1):
            output += f"{i}. {step}\n"
        
        output += f"\n💡 Suggested Approach:\n{result['suggested_approach']}\n"
        output += f"\n📊 Total Steps: {result['reasoning_count']}"
        
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
    """
    Create a simple LangChain Tool instance for Kaelum.
    
    Returns:
        A LangChain Tool that can be used with agents
    
    Usage:
        from langchain.agents import initialize_agent, AgentType
        from kaelum.integrations.langchain_tool import create_kaelum_tool
        from kaelum import set_reasoning_model
        
        set_reasoning_model(provider="ollama", model="qwen2.5:7b")
        
        kaelum_tool = create_kaelum_tool()
        agent = initialize_agent([kaelum_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    """
    try:
        from langchain.tools import Tool
    except ImportError:
        raise ImportError("LangChain not installed. Install with: pip install langchain")
    
    def kaelum_func(query: str) -> str:
        result = kaelum_enhance_reasoning(query=query, domain="general")
        
        output = "Reasoning Steps:\n"
        for i, step in enumerate(result["reasoning_steps"], 1):
            output += f"{i}. {step}\n"
        output += f"\nSuggested Approach: {result['suggested_approach']}"
        
        return output
    
    return Tool(
        name="KaelumReasoning",
        func=kaelum_func,
        description=(
            "Enhances reasoning for complex questions by breaking them down into "
            "logical steps. Use for math, logic, multi-step problems. "
            "Input: the question needing reasoning enhancement."
        )
    )
