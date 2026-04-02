##### Theory 

# Los modelos SOTA manejan una forma de poner sus respuestas en un formato que queramos (específico)
# Por default Langchain usa esto - la forma nativa de los modelos
# Si no lo soporta usa Tool calling strategy
# --- PydanticModel, Dataclass, Json, etc.

from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field # Structured datascheme. Automatic datatype validation

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
#from tavily import TavilyClient

#tavily = TavilyClient()

###########################################################
# Tool: Function that an LLM can excecute
# We can implement the function (API, Database, etc)
# El docstring lo conoce el LLM para entender cuándo usarlo
# Type hinting y nombre de variables también es importante

# def search(query: str) -> str:
#     """Search the web for the given query and return the results.

#     Args:
#         query: The search query string to look up.

#     Returns:
#         A string containing the search results.
#     """
#     print(f'Searching {query}')
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agente"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer") # If there's no source list it takes empty list

llm = ChatOpenAI(model = 'gpt-5')
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools, response_format=AgentResponse) 

def main():
    print("Hello!")
    result = agent.invoke({
        "messages": HumanMessage(
            content="Search for 3 job postings in AI/Data Science in Bogotá Colombia or with possibility of working from Bogotá Colombia on Linkedin. Give me the details.")})
    print(result)

if __name__ == '__main__':
    main()