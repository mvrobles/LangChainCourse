import os 
from dotenv import load_dotenv

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # identity function + we can add new parameters

load_dotenv()

print('Initializing components...')

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model = 'gpt-4')

vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'],
                                  embedding=embeddings)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}) # Everytime we search we get the top 3 docs

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)

def format_docs(docs): 
    """Format retrieved docs into one string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_langchain(query:str)->str:
    """Manually retrieves documents, format them and generate the response
    
    LIMITATIONS:
    - Manual step by step 
    - Hard debug/trace
    """
    
    docs = retriever.invoke(query) # usa pinecone para encontrar los top k documents
    context = format_docs(docs) # Pasarlo a string
    messages = prompt_template.format_messages(
        context = context, # Retrieved docs
        question = query # Initial prompt
    )
    response = llm.invoke(messages)
    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain usign LCEL (Langchain Expression Language)

    Advantages: 
    - Easy with pipe operator in LCEL
    - Everything in one place - the in LangSmith 
    - Less code
    - Reusable: Chain can be saved, shared and composed with other chains
    """
    # Prompt_template needs 2 parameters. 
    # We need to take the output retriever | format_docs and add another parameter (prompt/query)
    # itemgetter[str]("question") it's getting the question
    # The input chain is a dictionary {"question": "question..."}
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs, # Adding the second parameter context on top on the "question" that we already have
        )
        # Input: dictionary with question and context
        | prompt_template   # Template with user question and context
        | llm               # Invoke the llm with the prompt
        | StrOutputParser() # Get the response. Access the reponse.content
    )

    return retrieval_chain


if __name__ == '__main__':
    print("Starting")

    query = "What is Pinecone in Machine Learning"

    ## Option 0: Raw invocation without RAG
    print("-"*20)
    print(llm.invoke([HumanMessage(content=query)]).content)

    ## Option 1: Invocation without langchain
    print("-"*20)
    print(retrieval_chain_without_langchain(query))

    ## Option 2: Invocation with langchain
    print("-"*20)
    retrieval = create_retrieval_chain_with_lcel()
    print(retrieval.invoke({"question": query}))
