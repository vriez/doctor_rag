# llama-index-0.9.44

import os
import sys
import time
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import google.generativeai
from neo4j import exceptions
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

from llama_index.core import load_index_from_storage
from utils import dataset
# from anti_woke import *


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

start_from = int(sys.argv[1])
go_until = int(sys.argv[2])
print("start ", start_from, " till ", go_until)
# sys.exit(0)

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit s
llm = Gemini(temperature=0.0)
embedding_llm = GeminiEmbedding(model="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 2048

space_name = "doctor_rag_continuous"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg
database = "neo4j"

# 512
# username = "neo4j"
# chunk_size = Settings.chunk_size
# password = "news-rocks-subsystem"
# url = "bolt://REDACTED_IP:7687"
# # bolt+s://7f12b126f773205cdfea28a0ad768638.neo4jsandbox.com:7687

# # 2048 index
# username = "neo4j"
# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# # bolt+s://68db8edf1c2e12a3cc7f7860fe28d770.neo4jsandbox.com:7687

# # 2048 incremental
# username = "neo4j"
# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# db_id = "aa71f7f54748577d4ac173a4462cd074"
# # # bolt+s://aa71f7f54748577d4ac173a4462cd074.bolt.neo4jsandbox.com:443

# 2048 multithreaded index
username = "neo4j"
password = "barriers-brush-stocking"
url = "bolt://REDACTED_IP:7687"
# bolt+s://c8ac89364ecd0581662c26ca8fcd869e.neo4jsandbox.com:7687

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)#, persist_dir=f'./storage_graph_c8ac89364ecd0581662c26ca8fcd869e__2048')

df = pd.read_csv("sentences_syn.csv")
# df.reset_index(drop=True)
df["size"] = df["text"].str.len()
nodes = dataset(df, Settings.chunk_size)

                                               
kg_index_f = KnowledgeGraphIndex.from_documents(
    [],
    storage_context=storage_context,
    max_triplets_per_chunk=280,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    # show_progress=True,
    include_embeddings=True,
    verbose=True
)

kg_index_f.storage_context.persist(persist_dir=f'./storage_graph_c8ac89364ecd0581662c26ca8fcd869e__2048')

                                                                                              
def extract_triplets(node):
    triplets = kg_index_f._llm_extract_triplets(node.text)
    return list(set(triplets)), [node]

def process_node(node):
    # print("process_node: ", node)
    triplets = []
    try:
        triplets, node = extract_triplets(node)
        # return triplets
    # except Exception as e:
    except (exceptions.ServiceUnavailable, exceptions.TransientError) as e:
        time.sleep(10)
        triplets, node = extract_triplets(node)
        # return triplets                                            
    except (google.generativeai.types.generation_types.StopCandidateException, google.generativeai.types.generation_types.BlockedPromptException) as e:

        left_node, right_node = node.split()

        if len(left_node.text) > 0:
            left_triplets = process_node(left_node)
        else:
            left_triplets, left_node = []

        if len(right_node.text) > 0:
            right_triplets = process_node(right_node)
        else:
            right_triplets = []
        triplets = (left_triplets + right_triplets)
    return triplets
    

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

    left_node = Document(text=left_text, metadata=left_metadata)
    right_node = Document(text=right_text, metadata=right_metadata)

    return [left_node, right_node]

setattr(Document, "split", split)


unprocessed = []
triplets_list = []
def triplet_extractor(text, metadata):
    global pbar
    # doc = Document(text=text, metadata=metadata)
    # triplets = process_node(doc)
    # if triplets == []:
    #     unprocessed.append(metadata)
    # else:
    #     tag = f'{metadata["source"]}__{metadata["start"]}__{metadata["end"]}'
    #     triplets_list.append({"triplets": triplets, "id": tag})
    try:
        triplets = kg_index_f._llm_extract_triplets(text)
        tag = f'{metadata["source"]}__{metadata["start"]}__{metadata["end"]}'
        triplets_list.append({"triplets": triplets, "id": tag})
    except google.generativeai.types.generation_types.BlockedPromptException as e:
        # print(f"FAIL BlockedPromptException for {text}")
        # print(e)
        unprocessed.append(metadata)
        triplets = []
    except google.generativeai.types.generation_types.StopCandidateException as e:
        # print(f"FAIL StopCandidateException for {text}")
        # print(e)
        unprocessed.append(metadata)
        triplets = []
    except (exceptions.ServiceUnavailable, exceptions.TransientError) as e:
        time.sleep(10)
        # print(e)
        triplets = triplet_extractor(text, metadata)
    except Exception as e:
        # print(f"FAIL Exception for {text}")
        unprocessed.append(metadata)
        # print(e)
        time.sleep(3)
        triplets = []
    pbar.update(1)
    return triplets

nodes = nodes[start_from: go_until]
with tqdm(total=len(nodes)) as pbar:

    kg_index = KnowledgeGraphIndex.from_documents(
        nodes,
        storage_context=storage_context,
        max_triplets_per_chunk=280,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        # show_progress=True,
        kg_triplet_extract_fn=triplet_extractor,
        include_embeddings=True,
        verbose=True
    )

# kg_index.storage_context.persist(persist_dir=f'./storage_graph_bulk_25__{Settings.chunk_size}')

pd.DataFrame(unprocessed).to_csv(f"unprocessed_data__{start_from}_{go_until}.csv", index=None)

flat_data = [
    {'subject': triplet[0], 'relation': triplet[1], 'object': triplet[2], 'id': item['id']}
    for item in data for triplet in item['triplets']
]

# Convert to DataFrame
pd.DataFrame(flat_data).to_csv(f"triplets_data__{start_from}_{go_until}.csv", index=None)

# # kg_index.persist(persist_path="knowledge_graph.json")
# kg_index = load_index_from_storage(storage_context=storage_context)

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
