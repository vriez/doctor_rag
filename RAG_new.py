# llama-index-0.9.44

import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.schema import Node, Document
from IPython.display import Markdown, display
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
    StorageContext,
)
# from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import load_index_from_storage
# from anti_woke import *

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit s
llm = Gemini(temperature=0.0)
embedding_llm = GeminiEmbedding(model="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 2048

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], ["relationship"]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg
database = "neo4j"

# 512
# username = "neo4j"
# chunk_size = Settings.chunk_size
# password = "mints-indication-topic"
# url = "bolt://54.146.178.190:7687"
# bolt+s://d1d75140b2b1d7fd08d143a30d6c2730.neo4jsandbox.com:7687

# 1536
# username = "neo4j"
# chunk_size = Settings.chunk_size
# password = "letterhead-butters-clips"
# url = "bolt://44.220.84.232:7687"
# bolt+s://c5ddd8a8c31c239cc0ba42fe96f5bb17.neo4jsandbox.com:7687

# 2048 contiguous
username = "neo4j"
password = "officials-shapes-masks"
url = "bolt://3.235.103.212:7687"
db_id = "a59005a074b6b1e95f65b0b90df3e003"
# bolt+s://a59005a074b6b1e95f65b0b90df3e003.neo4jsandbox.com:7687


# 512 index
# username = "neo4j"
# password = "rear-alerts-manufacturers"
# url = "bolt://3.239.62.109:7687"



df = pd.read_csv("sentences_syn.csv")
# df.reset_index(drop=True)
df["size"] = df["text"].str.len()
print(df.dtypes)
# df["fname"] = df["fname"].str
# print(df.fname.unique())
# df = df[ df["fname"].str == 'e63c3c2f506e63c49a002a5e3ead8934' ]

documents = []
nodes = []
files = df.groupby("fname")

for f_name, f_content in files:
    chunk = ""
    start = None  # track the start from the group index
    # if f_name != "e63c3c2f506e63c49a002a5e3ead8934":
    #     continue

    # print(f_name)

    for i, row in f_content.iterrows():
        content = row.text.lower()
        size = len(content)
        # print()
        if size > Settings.chunk_size:
            # Handle single-row chunks directly
            metadata = {"source": f_name, "block_size": Settings.chunk_size, "size": size, "start": i+1, "end": i + 1}
            doc = Document(text=content.strip(), metadata=metadata)
            documents.append(doc)

            node = Node(text=content.strip(), metadata=metadata)
            nodes.append(node)
            start = i # Update start for potential subsequent multi-row chunks
            continue

        elif len(chunk) + size > Settings.chunk_size:
            # Handle multi-row chunk creation
            metadata = {"source": f_name, "block_size": Settings.chunk_size, "size": len(chunk), "start": start+1, "end": i}
            doc = Document(text=chunk.strip(), metadata=metadata)
            documents.append(doc)

            node = Node(text=chunk.strip(), metadata=metadata)
            nodes.append(node)
            chunk = ""
            start = i  # Update start for the next chunk
            continue

        else:
            # Accumulate text for multi-row chunks
            chunk += " " + content
            start = start or i
            continue

# print(len(lens), max(lens), min(lens), sum(lens)/len(lens))

# for metadata, text in texts:
#     doc = Document(text=text, metadata=metadata)
#     documents.append(doc)


graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# for i, doc in enumerate(documents):
#     if i == 221:
#         print(doc.text)

# documents = documents[:220+76+100]

kg_index = KnowledgeGraphIndex(
    [],
    storage_context=storage_context,
    max_triplets_per_chunk=80,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)


def extract_triplets(node):
    triplets = kg_index._extract_triplets(node.text)
    return list(set(triplets)), [node]

def process_node(node):

    try:
        triplets, node = extract_triplets(node)
        return triplets, node
    except Exception as e:

        start = node.metadata.get("start")
        end = node.metadata.get("end")
        
        mid = start + (end - start) // 2

        left_text = " ".join(df.iloc[start:mid, 2].tolist()).lower().strip()
        right_text = " ".join(df.iloc[mid:end, 2].tolist()).lower().strip()

        
        left_metadata = metadata.copy()
        left_metadata.update({"block_size": Settings.chunk_size, "size": len(left_text)+1, "start": start, "end": mid})
        
        right_metadata = metadata.copy()
        right_metadata.update({"block_size": Settings.chunk_size, "size": len(right_text)+1, "start": mid+1, "end": end})
        
        left_node = Node(text=left_text, metadata=left_metadata)
        right_node = Node(text=right_text, metadata=right_metadata)

        # print("L: ", left_node.text)
        # print()
        # print("R: ", right_node.text)
        # print()
        # print()

        if len(left_text) > 0:
            left_triplets, left_node = process_node(left_node)
        else:
            left_triplets, left_node = [], []

        if len(right_text) > 0:
            right_triplets, right_node = process_node(right_node)
        else:
            right_triplets, right_node = [], []

        return left_triplets + right_triplets, left_node + right_node
    
unsafe = []

for node in tqdm(nodes, total=len(nodes)):
    triplets, nodes = process_node(node)
    if triplets == []:
        unsafe.append(nodes)
    for tplt, node in zip(triplets, nodes):
        kg_index.upsert_triplet_and_node(triplet=tplt, node=node, include_embeddings=True)
        

# kg_index.persist(persist_path="knowledge_graph.json")
# kg_index = load_index_from_storage(storage_context=storage_context)
kg_index.storage_context.persist(persist_dir=f'./storage_graph_iter_/{Settings.chunk_size}')

query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

response = query_engine.query(
    "Tell me about the relationship between Vitamind D and Covid?",
)
display(Markdown(f"<b>{response}</b>"))
