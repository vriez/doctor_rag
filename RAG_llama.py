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
    SimpleDirectoryReader,
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

# os.environ["NEBULA_USER"] = "root"
# os.environ["NEBULA_PASSWORD"] = "nebula"
# os.environ["NEBULA_ADDRESS"] = "127.0.0.1:9669"

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], ["relationship"]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

# graph_store = NebulaGraphStore(space_name=space_name, edge_types=edge_types, rel_prop_names=rel_prop_names, tags=tags)
# username = "neo4j"
# password = "highways-greenwich-store"
# url = "bolt://44.204.173.145:7687"
# database = "neo4j"

username = "neo4j"
password = "password"
url = "neo4j://localhost:7687"
database = "neo4j"

data_folder = Path("/home/vitor/Downloads/converted_treated-20240308T134901Z-001")

data_files = data_folder.rglob("*.txt")

df = pd.read_csv("sentences.csv")
df.reset_index(drop=True)
df["size"] = df["text"].str.len()

chunk_size = 512
chunks = []
documents = []
files = df.groupby("fname")
for f_name, f_content in files:
    chunk = ""
    for i, row in f_content.iterrows():
        size = len(row.text)
        if size > chunk_size:
            chunks.append(row.text)
            metadata = {"source": f_name, "created_at": "2024-03-21"}
            doc = Document(text=row.text, metadata=metadata)
            documents.append(doc)
            continue
        elif len(chunk) + size > chunk_size:
            chunks.append(chunk)
            metadata = {"source": f_name, "created_at": "2024-03-21"}
            doc = Document(text=chunk, metadata=metadata)
            documents.append(doc)
            chunk = ""
        else:
            chunk += " " + row.text

# for c in chunks:
# 	print(len(c))
# 	metadata = {"source": fname.stem, "created_at": "2024-03-21"}
# 	doc = Document(
#         text=c,
#         metadata=metadata,
#         # source=fname.stem
#         # get_doc_id=fname.stem,
#     )


#     # get_doc = lambda self: fname.stem
#     # setattr(doc, 'get_doc_id', get_doc_id)

#     # doc.get_doc_id = get_doc

#     documents.append(doc)

# documents = []
# for fname in data_files:

#     with open(fname, "r") as fd:
#         data = fd.read()

#     metadata = {"source": fname.stem, "created_at": "2024-03-21"}
#     doc = Document(
#         text=data,
#         metadata=metadata,
#         # source=fname.stem
#         # get_doc_id=fname.stem,
#     )


#     # get_doc = lambda self: fname.stem
#     # setattr(doc, 'get_doc_id', get_doc_id)

#     # doc.get_doc_id = get_doc

#     documents.append(doc)

# print(documents)

# splitter = SentenceSplitter(
#     chunk_size=1024,
#     chunk_overlap=20,
# )
# nodes = splitter.get_nodes_from_documents(documents)
# # nodes = slides_parser.get_nodes_from_documents(documents)
# for i, node in enumerate(nodes):
#     get_doc = lambda self: i
#     # setattr(nodes[i], 'get_doc_id', get_doc)
#     nodes[i].get_doc_id = get_doc

# print(dir(nodes[-1]))

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

for doc in tqdm(documents, total=len(documents)):
    # if doc.metadata["source"] == "a0590e4a893c1e0c66eb19d53361cc60":
    #     print(doc)
    try:
        kg_index = KnowledgeGraphIndex.from_documents(
            documents=[doc],
            storage_context=storage_context,
            max_triplets_per_chunk=60,
            service_context=service_context,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True
        )
    except:
        print("FAIL: ", doc.metadata, "\n", doc)
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
