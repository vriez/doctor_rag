# llama-index-0.9.44

import os
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import google.generativeai
from multiprocessing import Pool
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from IPython.display import Markdown, display
from llama_index.core.schema import Node, Document
from llama_index.core import load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
    StorageContext,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from utils import dataset

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

space_name = "doctor_rag"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg
database = "neo4j"

# 2048 incremental
username = "neo4j"
password = "REDACTED_NEO4J_PASSWORD"
url = "bolt://REDACTED_IP:7687"
db_id = "aa71f7f54748577d4ac173a4462cd074"
# # bolt+s://aa71f7f54748577d4ac173a4462cd074.bolt.neo4jsandbox.com:443

# # 2048 index slice
# username = "neo4j"
# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# db_id = "7d3d267a81aee7b921190a2f09ea292f"
# # bolt+s://7d3d267a81aee7b921190a2f09ea292f.neo4jsandbox.com:7687

# # 2048 contiguous
# username = "neo4j"
# password = "officials-shapes-masks"
# url = "bolt://REDACTED_IP:7687"
# db_id = "bcd95aaad62b7b424b7f0675feac7185"
# # bolt+s://bcd95aaad62b7b424b7f0675feac7185.neo4jsandbox.com:7687


df = pd.read_csv("sentences_syn.csv")
# df.reset_index(drop=True)
df["size"] = df["text"].str.len()

nodes = dataset(df, Settings.chunk_size)


def split(node):
    start = node.metadata.get("start")
    end = node.metadata.get("end")

    mid = start + (end - start) // 2

    left_text = " ".join(df.iloc[start:mid, 2].tolist()).lower().strip()
    right_text = " ".join(df.iloc[mid:end, 2].tolist()).lower().strip()

    left_metadata = node.metadata.copy()
    left_metadata.update(
        {
            "block_size": Settings.chunk_size,
            "size": len(left_text) + 1,
            "start": start,
            "end": mid,
        }
    )

    right_metadata = node.metadata.copy()
    right_metadata.update(
        {
            "block_size": Settings.chunk_size,
            "size": len(right_text) + 1,
            "start": mid + 1,
            "end": end,
        }
    )

    left_node = Node(text=left_text, metadata=left_metadata)
    right_node = Node(text=right_text, metadata=right_metadata)

    return [left_node, right_node]


setattr(Node, "split", split)

indices = []
track = []
docs = []
counter = 0
processed = []

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
storage_context.persist(persist_dir=f"./storage_graph_doc_iter__{Settings.chunk_size}")


# # documents = documents[85:]
# while counter < 6:
#     print("it: ", counter)
#     for i, doc in tqdm(enumerate(documents), total=len(documents)):
#         try:
#             kg_index = KnowledgeGraphIndex.from_documents(
#                 [doc],
#                 storage_context=storage_context,
#                 max_triplets_per_chunk=240,
#                 space_name=space_name,
#                 edge_types=edge_types,
#                 rel_prop_names=rel_prop_names,
#                 tags=tags,
#                 show_progress=True,
#                 include_embeddings=True,
#             )
#             processed.append(doc.metadata)
#         except google.generativeai.types.generation_types.BlockedPromptException as e:
#             print(e)
#             docs.append(doc)
#             track.append(doc.metadata)
#         except google.generativeai.types.generation_types.StopCandidateException as e:
#             docs.extend(doc.split())
#             print("splitted documents ", i)
#             track.append(doc.metadata)
#         except Exception as e:
#             print(e)
#             time.sleep(3)

#     documents = docs
#     docs = []
#     counter += 1

# print(track)
# print(indices)
# counter += 1
# pd.DataFrame(indices).to_csv(f"indices__{start_from}_{go_until}__{counter}.csv", index=None)

# kg_index = KnowledgeGraphIndex.from_documents(
#     documents[:5],
#     storage_context=storage_context,
#     max_triplets_per_chunk=240,
#     space_name=space_name,
#     edge_types=edge_types,
#     rel_prop_names=rel_prop_names,
#     tags=tags,
#     # show_progress=True,
#     include_embeddings=True,
# )


kg_index = KnowledgeGraphIndex.from_documents(
    [],
    storage_context=storage_context,
    max_triplets_per_chunk=240,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    # show_progress=True,
    include_embeddings=True,
)

for node in nodes[:5]:
    triplets = kg_index._extract_triplets(node.text)
    # print(triplets)
    triplets = set(triplets)
    for tplt in triplets:
        kg_index.upsert_triplet_and_node(
            triplet=tplt, node=node, include_embeddings=True
        )


# kg_index.persist(persist_path="knowledge_graph__index.json")
# kg_index = load_index_from_storage(storage_context=storage_context)
kg_index.storage_context.persist(
    persist_dir=f"./storage_graph_doc_iter__{Settings.chunk_size}"
)

# query_engine = kg_index.as_query_engine(
#     include_text=True,
#     response_mode="tree_summarize",
#     embedding_mode="hybrid",
#     similarity_top_k=5,
# )

# response = query_engine.query(
#     "Tell me about the relationship between Vitamind D and Covid?",
# )
# display(Markdown(f"<b>{response}</b>"))


graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
)
