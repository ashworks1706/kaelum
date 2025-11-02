"""KaelumAI + LangChain Integration - Customer Service Agent Example

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
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Optional

# ============================================================================
# Mock Customer Service Tools (simulating realistic agent environment)
# ============================================================================


@tool
def get_order_status(order_id: str) -> str:
    """
    Retrieves the current status of a customer order.
    
    Args:
        order_id: The unique order identifier (e.g., ORD-12345)
    
    Returns:
        Order status information including shipping details
    """
    # Mock database lookup
    mock_orders = {
        "ORD-12345": "Order shipped on Dec 1, 2024. Expected delivery: Dec 5, 2024. Tracking: TRK789456",
        "ORD-67890": "Order processing. Payment confirmed. Will ship within 24 hours.",
    }
    return mock_orders.get(order_id, f"Order {order_id} not found in system.")


@tool
def check_product_availability(product_id: str, quantity: int = 1) -> str:
    """
    Checks if a product is in stock and available for purchase.
    
    Args:
        product_id: The product SKU or ID (e.g., PROD-001)
        quantity: Number of units needed (default: 1)
    
    Returns:
        Availability status and stock information
    """
    # Mock inventory check
    mock_inventory = {
        "PROD-001": {"name": "Wireless Mouse", "stock": 150, "price": 29.99},
        "PROD-002": {"name": "USB-C Cable", "stock": 0, "price": 12.99},
        "PROD-003": {"name": "Laptop Stand", "stock": 45, "price": 49.99},
    }
    
    if product_id in mock_inventory:
        product = mock_inventory[product_id]
        if product["stock"] >= quantity:
            return f"✓ {product['name']} is in stock. Available: {product['stock']} units. Price: ${product['price']}"
        else:
            return f"✗ {product['name']} is out of stock. Expected restock: Dec 10, 2024"
    return f"Product {product_id} not found."


@tool
def calculate_refund_amount(order_id: str, return_reason: str, days_since_purchase: int) -> str:
    """
    Calculates the refund amount for a product return based on company policy.
    Use kaelum_reasoning for complex refund calculations involving discounts, taxes, or prorations.
    
    Args:
        order_id: The order ID being returned
        return_reason: Reason for return (defective, wrong_item, changed_mind, etc.)
        days_since_purchase: Number of days since the original purchase
    
    Returns:
        Refund calculation details and eligibility
    """
    # Mock refund policy logic
    mock_orders = {
        "ORD-12345": {"total": 89.97, "tax": 7.20, "shipping": 5.99},
        "ORD-67890": {"total": 129.99, "tax": 10.40, "shipping": 0.00},
    }
    
    if order_id not in mock_orders:
        return f"Order {order_id} not found."
    
    order = mock_orders[order_id]
    
    if days_since_purchase > 30:
        return "❌ Refund denied: Return window (30 days) has expired."
    
    if return_reason == "defective":
        refund = order["total"] + order["shipping"]  # Full refund including shipping
        return f"✓ Full refund approved: ${refund:.2f} (defective item policy)"
    elif days_since_purchase <= 14:
        refund = order["total"]  # Refund without shipping
        return f"✓ Refund approved: ${refund:.2f} (shipping non-refundable)"
    else:
        refund = order["total"] * 0.85  # 15% restocking fee
        return f"✓ Partial refund: ${refund:.2f} (15% restocking fee applied after 14 days)"


@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the company knowledge base for policies, procedures, and FAQ answers.
    
    Args:
        query: Search query (e.g., "return policy", "warranty information")
    
    Returns:
        Relevant knowledge base article or policy
    """
    # Mock knowledge base
    kb = {
        "return policy": "Returns accepted within 30 days. Items must be unused. Refunds processed in 5-7 business days.",
        "warranty": "All electronics come with 1-year manufacturer warranty. Extended warranties available at checkout.",
        "shipping": "Free shipping on orders over $50. Standard delivery: 3-5 business days. Express available.",
        "price match": "We match competitor prices within 7 days of purchase. Requires proof of lower price.",
    }
    
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return f"📄 Knowledge Base: {value}"
    
    return "No matching policy found. Please consult the full policy documentation."


@tool
def create_support_ticket(customer_email: str, issue_summary: str, priority: str = "normal") -> str:
    """
    Creates a support ticket for issues requiring human agent escalation.
    
    Args:
        customer_email: Customer's email address
        issue_summary: Brief description of the issue
        priority: Ticket priority (low, normal, high, urgent)
    
    Returns:
        Ticket confirmation with ticket ID
    """
    import random
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    
    return f"✓ Support ticket created: {ticket_id}\nEmail: {customer_email}\nPriority: {priority}\nStatus: Open\nExpected response time: {2 if priority == 'urgent' else 24} hours"


if __name__ == "__main__":
    
    # Default prompt (baseline)
    reasoning_system = """You are a reasoning assistant. Break down problems into clear, logical steps.
Present your reasoning as a numbered list."""

#     # Experiment 1: More structured reasoning
    reasoning_system = """You are a precise reasoning engine. For each problem:
1. Identify what is being asked
2. List known information
3. Break down the solution into logical steps
4. Verify each step before proceeding
Present your reasoning as a numbered list."""

#     # Experiment 2: Chain-of-thought emphasis
    reasoning_system = """You are an expert at step-by-step reasoning. Think through problems carefully:
- State assumptions explicitly
- Show your work for calculations
- Explain why each step follows from the previous
- Double-check your logic
Format your response as a numbered list of reasoning steps."""
    
#     # Experiment 3: Domain-specific (math focus)
    reasoning_system = """You are a mathematical reasoning assistant. When solving problems:
- Define all variables clearly
- Show all algebraic steps
- Verify calculations
- State the final answer clearly
Present reasoning as numbered steps."""
        
    # User template experiments
    reasoning_template = "{query}"
    reasoning_template = "Think step-by-step about this problem: {query}"
    reasoning_template = "Break down and solve: {query}"

    print(f"{'='*80}\n")
    
    set_reasoning_model(
        base_url="http://localhost:8000/v1",
          model="Qwen/Qwen2.5-1.5B-Instruct",
    # model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
        temperature=0.3,
        max_tokens=512,
        reasoning_system_prompt=reasoning_system,
        reasoning_user_template=reasoning_template,
    )
    
    reasoning_tool = KaelumReasoningTool()
    
    # All available tools for customer service agent
    tools = [
        reasoning_tool,
        get_order_status,
        check_product_availability,
        calculate_refund_amount,
        search_knowledge_base,
        create_support_ticket,
    ]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)
    
    # System instruction - Kaelum as transparent reasoning acceleration
    system_instruction = """You are a helpful customer service AI agent with access to company tools.

**Your Available Tools:**
- get_order_status: Check order shipping and delivery status
- check_product_availability: Verify inventory and pricing  
- calculate_refund_amount: Process standard refund calculations
- search_knowledge_base: Look up company policies and FAQs
- create_support_ticket: Escalate issues to human agents
- kaelum_reasoning: Enhanced reasoning engine (internal cognitive tool)

**About kaelum_reasoning:**
This is your internal reasoning enhancement system. When you encounter a problem that requires:
- Breaking down into multiple logical steps
- Careful mathematical calculation
- Verification of your logic before committing to an answer
- Multi-variable analysis with dependencies

...you can offload that cognitive work to kaelum_reasoning. It acts as an extension of your thinking process - providing you with verified step-by-step breakdowns that have been checked for logical consistency and mathematical accuracy.

Think of it as your "slow thinking" mode - when you need to be absolutely certain, pause and use kaelum_reasoning to work through the problem methodically. The output will give you verified reasoning steps you can trust.

**How to work effectively:**
1. Read the customer's question carefully
2. If it's straightforward (order lookup, simple policy check), use the direct tools
3. If it requires complex reasoning or calculation, engage kaelum_reasoning first to work through the logic
4. Use other tools to fetch any needed data
5. Synthesize everything into a clear, helpful response

You don't need to explain which tools you're using to the customer - just provide accurate, professional service. kaelum_reasoning is your internal reasoning scaffold, invisible to them.

Always be professional, empathetic, and accurate in your customer interactions."""
    
    # Example customer service scenarios
    queries = [
        # Simple query - should NOT use kaelum_reasoning
        "What's the status of order ORD-12345?",
        
        # Complex query - SHOULD use kaelum_reasoning
        "I bought 3 items 20 days ago for $89.97 including $7.20 tax. If I return 2 items but keep 1, and there's a 15% restocking fee after 14 days, what's my refund?",
        
        # Policy query - use knowledge_base
        "What's your return policy?",
    ]
    
    # Test query - complex calculation that should trigger kaelum_reasoning
    query = queries[1]  # Use the complex one to show Kaelum's value
    
    print(f"\n{'='*80}")
    print(f"[TEST] Customer Service Agent with Kaelum Integration")
    print(f"{'='*80}")
    print(f"\n[INPUT] Query: {query}")
    
    response = llm_with_tools.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=query)
    ])
    
    # Process tool calls
    if response.tool_calls:
        print(f"\n[AGENT] Tool calls detected: {len(response.tool_calls)}")
        
        for i, tool_call in enumerate(response.tool_calls, 1):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n[TOOL {i}] {tool_name}")
            print(f"[ARGS] {tool_args}")
            
            # Execute the tool
            if tool_name == "kaelum_reasoning":
                result = reasoning_tool.invoke(tool_args)
                print(f"\n[KAELUM OUTPUT]")
                print(result)
            else:
                # Execute other tools
                tool_map = {
                    "get_order_status": get_order_status,
                    "check_product_availability": check_product_availability,
                    "calculate_refund_amount": calculate_refund_amount,
                    "search_knowledge_base": search_knowledge_base,
                    "create_support_ticket": create_support_ticket,
                }
                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    print(f"[RESULT] {result}")
        
        # Get final answer from agent after tool use
        print(f"\n{'='*80}")
        print(f"[AGENT] Synthesizing final response...")
        
        # Create a follow-up message with tool results
        tool_results = "\n\n".join([
            f"Tool: {tc['name']}\nResult: {reasoning_tool.invoke(tc['args']) if tc['name'] == 'kaelum_reasoning' else 'See above'}"
            for tc in response.tool_calls
        ])
        
        final_response = llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=query),
            HumanMessage(content=f"Based on the tool results:\n{tool_results}\n\nProvide a clear, professional customer service response.")
        ])
        
        print(f"\n[OUTPUT] Final Response:")
        print("-" * 80)
        print(final_response.content)
        print("-" * 80)
        
    else:
        print(f"\n[AGENT] No tool calls - direct response")
        print(f"\n[OUTPUT] {response.content}")
    
    print(f"\n{'='*80}")
    print(f"[END] Test complete")
    print(f"{'='*80}\n")