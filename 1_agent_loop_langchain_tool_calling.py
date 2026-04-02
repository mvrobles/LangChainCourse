from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from langchain_ollama import ChatOllama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# ------ Tools 

@tool 
def get_product_price(product: str)->float:
    """Look up the price of a product in the catalog.
    Available products: laptop, headphones, keyboard"""
    prices = {"laptop": 1299.99, 
              "headphones": 149.95, 
              "keyboard": 89.50}
    print(f"    >> Excecuting get_product_price {product}")
    return prices.get(product, 0)

@tool 
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a disccount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Excecuting apply_discount {price} - {discount_tier}")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier,0)
    return round(price*(1-discount/100),2)

# ----- Agent loop 

@traceable(name = 'LangChain Agent Loop')
def run_agent(question: str)->str:
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama: {MODEL}", temperature = 0)
    #llm = init_chat_model("openai:gpt-5", temperature = 0)

    # bind tools le dice al modelo qué tools están disponibles, la descripción y argumentos.
    llm_with_tools = llm.bind_tools(tools) # Todos los modelos con tool calling permiten tener esto
    
    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content = (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT_RULES - you must follow there exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price \n"
                "2. Only call apply_discount AFTER you have received"
                "a price from get_product_price - do NOT pass a made-up number. \n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier"
                "ask them which tier to use - do NOT assume one"
            )
        ),
        HumanMessage(
            content = question
        )
    ]
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n---- Iteration {iteration}")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls 

        if not tool_calls:
            print(f'Final answer: {ai_message.content}')
            return ai_message.content

        else: 
            tool_call = tool_calls[0]
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            tool_call_id = tool_call.get("id")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool {tool_name} nto found")
            
            observation = tool_to_use.invoke(tool_args)

            print(f"   Tool selected: {tool_name} with args {tool_args} with result {observation}.")

            messages.append(ai_message)
            messages.append(
                ToolMessage(content=str(observation), tool_call_id = tool_call_id) # 
             )

if __name__ == "__main__":
    print("Hello LangChain Agent")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount")