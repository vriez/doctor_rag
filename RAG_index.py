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
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
    StorageContext,
)

# start_from = int(sys.argv[1])
# go_until = int(sys.argv[2])

# print(start_from, go_until)

# sys.exit(0)

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
edge_types, rel_prop_names = ["relationship"], [
    "property"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg
database = "neo4j"

# 512
# username = "neo4j"
# password = "news-rocks-subsystem"
# url = "bolt://REDACTED_IP:7687"
# db_id = "7f12b126f773205cdfea28a0ad768638"
# # bolt+s://7f12b126f773205cdfea28a0ad768638.neo4jsandbox.com:7687

# # 2048 index
# username = "neo4j"
# password = "absences-pin-milligram"
# url = "bolt://REDACTED_IP:7687"
# db_id = "7d3d267a81aee7b921190a2f09ea292f"
# # bolt+s://7d3d267a81aee7b921190a2f09ea292f.neo4jsandbox.com:7687

# 2048 contiguous
username = "neo4j"
password = "officials-shapes-masks"
url = "bolt://REDACTED_IP:7687"
db_id = "a59005a074b6b1e95f65b0b90df3e003"
# bolt+s://a59005a074b6b1e95f65b0b90df3e003.neo4jsandbox.com:7687

indicea = [
    86,
    124,
    163,
    200,
    223,
    721,
    915,
    934,
    952,
    953,
    1068,
    1069,
    1071,
    1072,
    1073,
    1075,
    1076,
    1077,
    1078,
    1217,
    1233,
    1261,
    1267,
    1716,
    2319,
]

df = pd.read_csv("sentences_syn.csv")
# df.reset_index(drop=True)
df["size"] = df["text"].str.len()
# print(df.dtypes)
# df["fname"] = df["fname"].str
# print(df.fname.unique())
# df = df[ df["fname"].str == 'e63c3c2f506e63c49a002a5e3ead8934' ]

documents = []
nodes = []
files = df.groupby("fname")

for f_name, f_content in files:
    chunk = ""
    start = None  # track the start from the group index
    for i, row in f_content.iterrows():
        content = row.text.lower()
        size = len(content)
        # print()
        if size > Settings.chunk_size:
            # Handle single-row chunks directly
            metadata = {
                "source": f_name,
                "block_size": Settings.chunk_size,
                "size": size,
                "start": i + 1,
                "end": i + 1,
            }
            doc = Document(text=content.strip(), metadata=metadata)
            documents.append(doc)

            node = Node(text=content.strip(), metadata=metadata)
            nodes.append(node)
            start = i  # Update start for potential subsequent multi-row chunks
            continue

        elif len(chunk) + size > Settings.chunk_size:
            # Handle multi-row chunk creation
            metadata = {
                "source": f_name,
                "block_size": Settings.chunk_size,
                "size": len(chunk),
                "start": start + 1,
                "end": i,
            }
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

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# kg_index = KnowledgeGraphIndex(
#     [],
#     storage_context=storage_context,
#     max_triplets_per_chunk=80,
#     space_name=space_name,
#     edge_types=edge_types,
#     rel_prop_names=rel_prop_names,
#     tags=tags,
#     include_embeddings=True,
# )

def split(node):
    start = node.metadata.get("start")
    end = node.metadata.get("end")

    mid = start + (end - start) // 2

    left_text = " ".join(df.iloc[start:mid, 2].tolist()).lower().strip()
    right_text = " ".join(df.iloc[mid:end, 2].tolist()).lower().strip()

    left_metadata = metadata.copy()
    left_metadata.update(
        {
            "block_size": Settings.chunk_size,
            "size": len(left_text) + 1,
            "start": start,
            "end": mid,
        }
    )

    right_metadata = metadata.copy()
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

indices = [
    86,
    86 + 38,
    86 + 38 + 1, #
    86 + 38 + 1 + 147,
    86 + 38 + 1 + 147 + 67,
    86 + 38 + 1 + 147 + 67 + 116,
    86 + 38 + 1 + 147 + 67 + 116 + 1,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1, 
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1, #
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1 + 95,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1 + 95 + 1,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1 + 95 + 1 + 1,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1 + 95 + 1 + 1 + 10,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1 + 95 + 1 + 1 + 10 + 1,
    86 + 38 + 1 + 147 + 67 + 116 + 1 + 2 + 145 + 66 + 1 + 9 + 1 + 120 + 1 + 95 + 1 + 1 + 10 + 1 + 117 + 1
]

indices_1 = [
    86,
    456,
    587,
    801,
    897,
    909,
    1029,
    1030,
    1032,
    1036,
    1037,
    1176,
    1192,
    1194,
    1225,
    1342,
    1445,
    1822,
    1891,
    2057
]

indices_2 = [
    8,
    9,
    11,
    16,
    19,
    20,
    21
]

indices_3 = [
    1,
    2,
    6,
    7,
    12
]

indices_4 = [
    3,
    4,
    5,
    7,
]

indices_5 = [
    3
]

indices = []
indices.append(indices_1)
indices.append(indices_2)
indices.append(indices_3)
indices.append(indices_4)
indices.append(indices_5)

counter = 0

print("M: ", len(documents), len(indices), indices)
first = 0
print()
meta_docs = []
for d in documents:
    meta_docs.append(d.metadata)

pd.DataFrame(meta_docs).to_csv("original.csv", index=None)

# for indixes in indices:
    
#     replacements = {i: documents[i].split() for i in indixes}
#     print("i: ", [ l + first for l in indixes ])
#     sliced_documents = documents
#     for j, x in enumerate(documents):
#         if j not in replacements:
#             # sliced_documents.append(x)
#             pass
#         else:
#             sliced_documents.extend(replacements[j])
#     documents = sliced_documents
#     # docs.append(documents)

# # for k in docs:
# #     print(len(k))

# print(len(documents))
# meta_docs = []
# for d in documents:
#     meta_docs.append(d.metadata)

# pd.DataFrame(meta_docs).to_csv("processed.csv", index=None)
# print(len(docs))
# print(docs)
# sys.exit(0)

indices = []
track = []
docs = []
counter = 0
processed = []

# documents = documents[85:]
while counter < 6:
    print("it: ", counter)
    for i, doc in tqdm(enumerate(documents), total=len(documents)):
        try:
            kg_index = KnowledgeGraphIndex.from_documents(
                [doc],
                storage_context=storage_context,
                max_triplets_per_chunk=240,
                space_name=space_name,
                edge_types=edge_types,
                rel_prop_names=rel_prop_names,
                tags=tags,
                show_progress=True,
                include_embeddings=True,
            )
            processed.append(doc.metadata)
        except google.generativeai.types.generation_types.BlockedPromptException as e:
            print(e)
            docs.append(doc)
            track.append(doc.metadata)
        except google.generativeai.types.generation_types.StopCandidateException as e:
            docs.extend(doc.split())
            print("splitted documents ", i)
            track.append(doc.metadata)
        except Exception as e:
            print(e)
            time.sleep(3)
    
    documents = docs
    docs = []
    counter += 1

# print(track)
# print(indices)
# counter += 1
# pd.DataFrame(indices).to_csv(f"indices__{start_from}_{go_until}__{counter}.csv", index=None)

# kg_index.persist(persist_path="knowledge_graph__index.json")
# kg_index = load_index_from_storage(storage_context=storage_context)
kg_index.storage_context.persist(persist_dir=f'./storage_graph_index__{Settings.chunk_size}')

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
