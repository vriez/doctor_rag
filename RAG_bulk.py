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
from utils import dataset, dataset_overlap
# from anti_woke import *


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# start_from = int(sys.argv[1])
# go_until = int(sys.argv[2])
# print(f"python RAG_bulk.py {start_from*200} {(start_from+1) * 200} &")
# sys.exit(0)

# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit s
llm = Gemini(temperature=0.0)
embedding_llm = GeminiEmbedding(model="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 2048

space_name = "doctor_rag_continuous"
edge_types, rel_prop_names = ["relationship"], ["relationship"] 
tags = ["entity"]
database = "neo4j"
MAX_TRIPLETS = 320
# 512
# username = "neo4j"
# chunk_size = Settings.chunk_size
# password = "news-rocks-subsystem"
# url = "bolt://100.27.45.13:7687"
# # bolt+s://7f12b126f773205cdfea28a0ad768638.neo4jsandbox.com:7687

# # 2048 index
# username = "neo4j"
# password = "bureau-auto-adherences"
# url = "bolt://100.27.45.13:7687"
# # bolt+s://68db8edf1c2e12a3cc7f7860fe28d770.neo4jsandbox.com:7687

# 2048 incremental
username = "neo4j"
password = "nail-interface-necks"
url = "bolt://3.238.101.93:7687"
db_id = "aa71f7f54748577d4ac173a4462cd074"
# # bolt+s://aa71f7f54748577d4ac173a4462cd074.bolt.neo4jsandbox.com:443

# # 2048 multithreaded index
# username = "neo4j"
# password = "barriers-brush-stocking"
# url = "bolt://44.200.14.6:7687"
# # bolt+s://c8ac89364ecd0581662c26ca8fcd869e.neo4jsandbox.com:7687


# # 2048 multithreaded
# username = "neo4j"
# password = "machines-electrolytes-troop"
# url = "bolt://3.239.198.148:7687"
# # bolt+s://26d1b177537db8832f0d69488ed8fa41.neo4jsandbox.com:7687

# 2048 multithreaded index
# username = "neo4j"
# password = "accusation-tube-blueprints"
# url = "bolt://3.91.206.92:7687"
# # bolt+s://bcd95aaad62b7b424b7f0675feac7185.neo4jsandbox.com:7687

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_path = f'./storage_graph_{db_id}__2048'
storage_context = StorageContext.from_defaults(graph_store=graph_store)  #, persist_dir=f'./storage_graph_bcd95aaad62b7b424b7f0675feac7185__2048')

df = pd.read_csv("sentences_syn.csv")
# df.reset_index(drop=True)
df["size"] = df["text"].str.len()
# nodes = dataset(df, Settings.chunk_size)
nodes = dataset_overlap(df, Settings.chunk_size, 1)
print("nodes: ", len(nodes))
# udfs = []
# for p in Path(".").glob("unprocessed_data*.csv"):
#     udfs.append(pd.read_csv(p))

# udf = pd.concat(udfs)

# nodes = []
# fs = udf.groupby("source")
# for i, m in fs:
#     # print("M: ", i, m.shape)
#     for j,v in m.iterrows():
#         # print("J: ", i, j, v)
#         start = v["start"]
#         end = v["end"]
#         size = v["size"]
#         text = " ".join(df.iloc[start:end, 2].tolist()).lower()
#         print("J: ", size, dict(m), len(" ".join(df.iloc[start:end, 2].tolist()).lower()))
#         node = Document(text=text, metadata={"source": i, "size": size, "start": start, "end": end})
#         nodes.append(node)

# sliced_nodes = [ ]
# for n in nodes:
#     sliced_nodes.extend(n.split())

# nodes = sliced_nodes

kg_index_f = KnowledgeGraphIndex.from_documents(
    [],
    storage_context=storage_context,
    max_triplets_per_chunk=MAX_TRIPLETS,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    # show_progress=True,
    include_embeddings=True,
    verbose=True,
    timeout=100
)

# kg_index_f.storage_context.persist(persist_dir=f'./storage_graph_c8ac89364ecd0581662c26ca8fcd869e__2048')
kg_index_f.storage_context.persist(persist_dir=storage_path)
                                                                                              
def extract_triplets(node):
    triplets = kg_index_f._extract_triplets(node.text, node.metadata)
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
    # global pbar

    doc = Document(text=text, metadata=metadata)
    # triplets = process_node(doc)
    # if triplets == []:
    #     unprocessed.append(metadata)
    # else:
    #     tag = f'{metadata["source"]}__{metadata["start"]}__{metadata["end"]}'
    #     triplets_list.append({"triplets": triplets, "id": tag})
    try:
        # triplets = kg_index_f._llm_extract_triplets(text)
        triplets = process_node(doc)
        tag = f'{metadata["source"]}__{metadata["start"]}__{metadata["end"]}'
        triplets_list.append({"triplets": triplets, "id": tag})
    except google.generativeai.types.generation_types.BlockedPromptException as e:
        # print(f"FAIL BlockedPromptException for {text}")
        print(e)
        unprocessed.append(metadata)
        triplets = []
    except google.generativeai.types.generation_types.StopCandidateException as e:
        # print(f"FAIL StopCandidateException for {text}")
        print(e)
        unprocessed.append(metadata)
        triplets = []
    except (exceptions.ServiceUnavailable, exceptions.TransientError) as e:
        time.sleep(10)
        print(e)
        triplets = process_node(doc)
    except Exception as e:
        # print(f"FAIL Exception for {text}")
        unprocessed.append(metadata)
        print(e)
        time.sleep(3)
        triplets = []
    pbar.update(1)
    return triplets

# nodes = nodes[830:]
with tqdm(total=len(nodes)) as pbar:

    kg_index = KnowledgeGraphIndex.from_documents(
        nodes,
        storage_context=storage_context,
        max_triplets_per_chunk=MAX_TRIPLETS,
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

kg_index.storage_context.persist(persist_dir=storage_path)

pd.DataFrame(unprocessed).to_csv(f"f_unprocessed_data_{db_id}_1.csv", index=None)

flat_data = [
    {'subject': triplet[0], 'relation': triplet[1], 'object': triplet[2], 'id': item['id']}
    for item in triplets_list for triplet in item['triplets']
]

# Convert to DataFrame
pd.DataFrame(flat_data).to_csv(f"f_triplets_data_{db_id}.csv", index=None)

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
