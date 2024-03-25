# llama-index-0.9.44

import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
)
from llama_index import Document
from multiprocessing import Pool
from llama_index.llms import Gemini, Ollama
from IPython.display import Markdown, display
# LangChain supports many other chat models. Here, we're using Ollama
from llama_index.graph_stores import Neo4jGraphStore
from langchain_community.chat_models import ChatOllama
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings.google import GoogleUnivSentEncoderEmbedding
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm = Gemini(temperature=0.0)

# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)
# llm = Ollama(model="mistral", temperature=0.0)

embedding_llm = GeminiEmbedding(model="models/embedding-001")
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_llm,
)

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], ["relationship"]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

# 512
# username = "neo4j"
# chunk_size = 512
# password = "mints-indication-topic"
# url = "bolt://54.146.178.190:7687"
# database = "neo4j"


# 1536
username = "neo4j"
chunk_size = 1536
password = "letterhead-butters-clips"
url = "bolt://44.220.84.232:7687"
database = "neo4j"

# username = "neo4j"
# password = "password"
# url = "neo4j://localhost:7687"
# database = "neo4j"

df = pd.read_csv("sentences_syn.csv")
df["text"]
df.reset_index(drop=True)
df["size"] = df["text"].str.len()

chunks = []
documents = []
files = df.groupby("fname")
for f_name, f_content in files:
    chunk = ""
    for i, row in f_content.iterrows():
        content = row.text
        size = len(content)
        if size > chunk_size:
            chunks.append(content)
            metadata = {"source": f_name, "block_size": chunk_size, "size": len(chunk), "start": i, "end": i + 1}
            doc = Document(text=content.lower(), metadata=metadata)
            documents.append(doc)
            continue
        elif len(chunk) + size > chunk_size:
            chunks.append(chunk)
            metadata = {"source": f_name, "block_size": chunk_size, "size": len(chunk), "start": start, "end": i + 1}
            doc = Document(text=chunk.lower(), metadata=metadata)
            documents.append(doc)
            chunk = ""
            # start = i+1
            continue
        else:
            chunk += " " + content
        start = i


graph_store = Neo4jGraphStore(
    username=username, password=password, url=url
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

for doc in tqdm(documents, total=len(documents)):
    # if doc.metadata["source"] == "a0590e4a893c1e0c66eb19d53361cc60":
    #     print(doc)
    try:
        kg_index = KnowledgeGraphIndex.from_documents(
            documents=[doc],
            storage_context=storage_context,
            #max_triplets_per_chunk=max_triplets_per_chunk,
            service_context=service_context,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
            timeout=60,
        )
    except:
        print("FAIL: ", doc.metadata)
        print(doc.text)
        print()
        print()


# def process_document(doc):
#   try:
#       kg_index = KnowledgeGraphIndex.from_documents(
#           documents=[doc],
#           storage_context=storage_context,
#           max_triplets_per_chunk=60,
#           service_context=service_context,
#           space_name=space_name,
#           edge_types=edge_types,
#           rel_prop_names=rel_prop_names,
#           tags=tags,
#           include_embeddings=True
#       )
#   except Exception as e:
#       print(f"FAIL: {doc.metadata}, Error: {e}")

# num_processes = 8

# with Pool(processes=num_processes) as pool:
#   pool.map(process_document, documents)

# # install related packages, password is nebula by default
# %pip install ipython-ngql networkx pyvis
# %load_ext ngql
# %ngql --address 127.0.0.1 --port 9669 --user root --password <password>

# # Query some random Relationships with Cypher
# %ngql USE llamaindex;
# %ngql MATCH ()-[e]->() RETURN e LIMIT 10

# from llama_index.query_engine import KnowledgeGraphQueryEngine
# from llama_index.storage.storage_context import StorageContext
# from llama_index.graph_stores import NebulaGraphStore

# query_engine = KnowledgeGraphQueryEngine(
#     storage_context=storage_context,
#     service_context=service_context,
#     llm=llm,
#     verbose=True,
# )

# response = query_engine.query(
#     "Tell me about Peter Quill?",
# )
# display(Markdown(f"<b>{response}</b>"))
