import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 

# Vector database 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print('Ingesting...')
    loader = TextLoader('mediumblog1.txt')
    document = loader.load() # Langchain document

    print("Splitting...")
    # 1000 rule of thumb - depending on the context window of the model 
    # With SOTA models - higher
    text_splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap = 0)
    texts = text_splitter.split_documents(document)
    print(f"Created the chunks {len(texts)}.")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(texts,
                                       embeddings,
                                       index_name = os.environ['INDEX_NAME'])
    
    print("Finished")
