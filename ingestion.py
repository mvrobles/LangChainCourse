# Tavily will help us downloading the data that we need

import os
import ssl # 
import asyncio
from typing import Any, Dict, List

import certifi

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import log_header, log_info, Colors, log_success, log_error, log_warning

load_dotenv()

# Configuración SSL para hacer los request y descargar
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Rate limited - each vendor has different rate limits.
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10 # Handling rate limit
)

# Create vector store
vectorstore = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding = embeddings)
vectorstore.delete(delete_all=True) 

# Tavily
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth = 5, max_breath=20, max_pages = 1000)
tavily_crawl = TavilyCrawl()

async def add_batch(batch: List[Document], batch_num:int):
    try:
        await vectorstore.aadd_documents(batch)
        await asyncio.sleep(1)
        log_success(f"VectorStore Indexing: Successfully added batch number {batch_num}")

    except Exception as e:
        log_error(f"Vectors are indexing: Failed to add batch {batch_num} - {e}")
        return False
    return True

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(f"VectorStore Indexing: {len(batches)} batches to process")

    successful = 0
    for i, batch in enumerate(batches):
        try:
            await vectorstore.aadd_documents(batch)
            await asyncio.sleep(1)
            log_success(f"VectorStore Indexing: Batch {i+1}/{len(batches)} added")
            successful += 1
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed batch {i+1} - {e}")

    if successful == len(batches):
        log_success(f"All {successful}/{len(batches)} batches processed successfully")
    else:
        log_warning(f"VectorStore Indexing: Processed {successful}/{len(batches)} batches")

async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "Tavily Crawl: Starting", Colors.PURPLE
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://scikit-learn.org/stable/supervised_learning.html",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )
    
    all_docs = [Document(page_content=result['raw_content'], 
                         metadata = {"source": result['url']}) 
                         for result in res['results']]
    log_success(
        f"Tavily Crawl: Successfully crawled {len(all_docs)} URLs"
    )
    
    # Quiero pasarlo a PineconeVectorStore
    log_header("DOCUMENTATION CHUNKING PHASE")
    log_info(
        f"Text splitter: processing {len(all_docs)} URLs"
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Tavily Splitter: Creates {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    await index_documents_async(splitted_docs)

if __name__ == '__main__':
    print('--')
    asyncio.run(main())