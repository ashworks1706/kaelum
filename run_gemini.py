"""KaelumAI + Gemini - Reasoning Enhancement Demo"""

import os
from kaelum import set_reasoning_model, kaelum_enhance_reasoning
import google.generativeai as genai

set_reasoning_model(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:7b",
    temperature=0.3,
    max_tokens=512,
    max_reflection_iterations=0,
    use_symbolic_verification=True,
    use_factual_verification=False,
    rag_adapter=None,
)

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

kaelum_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="kaelum_enhance_reasoning",
            description="Enhances reasoning for complex problems with step-by-step breakdown",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "query": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Question needing reasoning enhancement"
                    ),
                    "domain": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Domain: math, logic, code, science, or general"
                    )
                },
                required=["query"]
            )
        )
    ]
)

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", tools=[kaelum_tool])

query = "If a train travels at 60 mph for 2.5 hours, then speeds up to 80 mph for another 1.5 hours, what's the total distance?"
print(f"\n💬 Question: {query}\n")

chat = model.start_chat()
response = chat.send_message(query)

if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    print("🧠 Gemini calling Kaelum...\n")
    
    result = kaelum_enhance_reasoning(
        query=function_call.args.get("query", query),
        domain=function_call.args.get("domain", "general")
    )
    
    print("✓ Reasoning steps:")
    for i, step in enumerate(result["reasoning_steps"], 1):
        print(f"   {i}. {step}")
    
    function_response = genai.protos.Part(
        function_response=genai.protos.FunctionResponse(
            name="kaelum_enhance_reasoning",
            response={"result": result}
        )
    )
    
    response = chat.send_message(function_response)
    print(f"\n📝 Gemini's Answer:\n{response.text}")
else:
    print(f"📝 Direct Answer:\n{response.text}")

