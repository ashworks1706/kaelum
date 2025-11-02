"""KaelumAI + LangChain Integration

Start vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model TinyLlama/TinyLlama-1.1B-Chat-v0.3 \
        --port 8000 \
        --gpu-memory-utilization 0.7 \
        --max-num-seqs 32 \
        --max-model-len 1024 \
        --chat-template "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}assistant: "
"""

from kaelum import set_reasoning_model
from kaelum.integrations.langchain_tool import KaelumReasoningTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    set_reasoning_model(
        base_url="http://localhost:8000/v1",
        model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
        temperature=0.3,
        max_tokens=512,
    )
    
    reasoning_tool = KaelumReasoningTool()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    llm_with_tools = llm.bind_tools([reasoning_tool])
    
    query = "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?"
    
    response = llm_with_tools.invoke([HumanMessage(content=query)])
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        reasoning = reasoning_tool.invoke(tool_call["args"])
        
        follow_up = f"""I used reasoning tools and got these steps:

{"\n".join(f"{i}. {step}" for i, step in enumerate(reasoning["reasoning_steps"], 1))}

Now provide the final answer to: {query}"""
        
        answer = llm.invoke([HumanMessage(content=follow_up)]).content
        print(f"Question: {query}\n")
        print("Reasoning:")
        for i, step in enumerate(reasoning["reasoning_steps"], 1):
            print(f"{i}. {step}")
        print(f"\nAnswer: {answer}")
    else:
        print(response.content)