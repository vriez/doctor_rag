import os
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import dataset
from neo4j import exceptions
from llama_index.core import Settings
from llama_index.core.schema import Node
from llama_index.llms.gemini import Gemini
from IPython.display import Markdown, display
from llama_index.core import load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
    StorageContext,
    ServiceContext,
)

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = Gemini(temperature=0.0)
embedding_llm = GeminiEmbedding(model="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 2048

space_name = "doctor_rag"
edge_types, rel_prop_names = ["relationship"], ["property"]
tags = ["entity"]

database = "neo4j"

# # 2048
# username = "neo4j"
# chunk_size = Settings.chunk_size
# password = "accusation-tube-blueprints"
# url = "bolt://3.91.206.92:7687"
# # bolt+s://bcd95aaad62b7b424b7f0675feac7185.neo4jsandbox.com:7687

# 2048 incremental
username = "neo4j"
password = "nail-interface-necks"
url = "bolt://3.238.101.93:7687"
db_id = "aa71f7f54748577d4ac173a4462cd074"
# bolt+s://aa71f7f54748577d4ac173a4462cd074.bolt.neo4jsandbox.com:443

df = pd.read_csv("sentences_syn.csv")

nodes = dataset(df, Settings.chunk_size)

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex(
    [],
    storage_context=storage_context,
    max_triplets_per_chunk=240,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
    timeout=60,
)


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


def extract_triplets(node):
    triplets = kg_index._extract_triplets(node.text)
    return list(set(triplets)), [node]


def process_node(node):

    try:
        triplets, node = extract_triplets(node)
        return triplets, node
    except Exception as e:
        print(e)
        left_node, right_node = node.split()

        if len(left_node.text) > 0:
            left_triplets, left_node = process_node(left_node)
        else:
            left_triplets, left_node = [], []

        if len(right_node.text) > 0:
            right_triplets, right_node = process_node(right_node)
        else:
            right_triplets, right_node = [], []

        return left_triplets + right_triplets, left_node + right_node


unsafe = []
start_time = time.time()
# for node in tqdm(nodes, total=len(nodes)):

index = 0
with tqdm(total=len(nodes)) as pbar:
    while index < len(nodes):
        node = nodes[index]

        try:
            triplets, sliced_nodes = process_node(node)
            index += 1
            pbar.update(1)
        except exceptions.ServiceUnavailable:
            time.sleep(10)
            continue
        except Exception as e:
            print(e)
            time.sleep(1)
            continue

        if triplets == []:
            unsafe.append(node)

        for tplt, node in zip(triplets, sliced_nodes):
            kg_index.upsert_triplet_and_node(
                triplet=tplt, node=node, include_embeddings=True
            )

end_time = time.time()

print(f"Processed all documents in {end_time - start_time} seconds")

kg_index.storage_context.persist(
    persist_dir=f"./storage_graph_iter_{Settings.chunk_size}"
)
