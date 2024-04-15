# llama-index-0.9.44

import os
import gc
import sys
import json
import time
import random
import logging
import pandas as pd
from utils import (
    dataset_whole,
    dataset_overlap,
    dataset,
)
from tqdm import tqdm
from pathlib import Path
import google.generativeai
from neo4j import exceptions
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


# Q&A
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.indices.loading import load_indices_from_storage
from llama_index.core.evaluation import FaithfulnessEvaluator
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


llm = Gemini(model_name="models/gemini-1.0-pro", temperature=0.0)
embedding_llm = GeminiEmbedding(model="models/embedding-001", temperature=0.0)
cypher_model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0)
qa_model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0)

evaluator = FaithfulnessEvaluator(llm=llm)

database = "neo4j"
username = "neo4j"

# # # index_d29e690203979119220cf60f40490e26_overlap_286__2048
# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# DB_ID = "d29e690203979119220cf60f40490e26"
# # # bolt+s://d29e690203979119220cf60f40490e26.neo4jsandbox.com:7687

# # index_7effdf391d1301b4dff0fdd97838e307_overlap_574__4096
# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# DB_ID = "7effdf391d1301b4dff0fdd97838e307"

# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# DB_ID = "69687aead0421f367f69aac6f1249cb2"

# password = "REDACTED_NEO4J_PASSWORD"
# url = "bolt://REDACTED_IP:7687"
# DB_ID = "8d02074acea413208355e0c16b66dc4e"
# overlap = sys.argv[3]
# EXP_TAG = "block"
# MAX_TRIPLETS = 286
# CHUNK_SIZE = 2048

password = sys.argv[1]
url = sys.argv[2]
DB_ID = sys.argv[3]
overlap = int(sys.argv[4])
EXP_TAG = sys.argv[5]
CHUNK_SIZE = int(sys.argv[6])
MAX_TRIPLETS = int(sys.argv[7])

print(
    "~ ",
    password,
    url,
    DB_ID,
    overlap,
    EXP_TAG,
    CHUNK_SIZE,
    MAX_TRIPLETS,
)
STORAGE_PATH = f"./v_storage_graph_{DB_ID}_{EXP_TAG}_{MAX_TRIPLETS}__{CHUNK_SIZE}"
SPACE_NAME = f"v_index_{DB_ID}_{EXP_TAG}_{MAX_TRIPLETS}__{CHUNK_SIZE}"

TRIPLET_FILE = Path(
    f"v_triplets_{DB_ID}_{EXP_TAG}_{MAX_TRIPLETS}__{CHUNK_SIZE}"
).with_suffix(".csv")
UNPROCESSED_FILE = Path(
    f"v_unprocessed_{DB_ID}_{EXP_TAG}_{MAX_TRIPLETS}__{CHUNK_SIZE}"
).with_suffix(".csv")
try:
    TRIPLET_FILE.unlink()
except:
    pass

try:
    UNPROCESSED_FILE.unlink()
except:
    pass

TRIPLET_FILE.touch()
UNPROCESSED_FILE.touch()

with open(TRIPLET_FILE, "a") as fd:
    fd.write("[\n")

with open(UNPROCESSED_FILE, "a") as fd:
    fd.write("[\n")

graph = Neo4jGraph(url=url, username=username, password=password, database="neo4j")

print("space_name: ", SPACE_NAME)
Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = CHUNK_SIZE

edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)
df = pd.read_csv("corpus.csv")
df["size"] = df["text"].str.len()

docs = dataset_overlap(df, CHUNK_SIZE, overlap)
# docs = dataset_whole(df)
print("nodes: ", len(docs))

kg_index_f = KnowledgeGraphIndex.from_documents(
    [],
    storage_context=storage_context,
    max_triplets_per_chunk=MAX_TRIPLETS,
    space_name=SPACE_NAME,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
    verbose=True,
    timeout=100,
)


def extract_triplets(node):
    for _ in range(5):
        try:
            triplets = kg_index_f._extract_triplets(node.text, node.metadata)
            break
        except:
            time.sleep(0.3)
    return list(set(triplets)), [node]


def process_node(node):
    # print("process_node: ", node)
    triplets = []
    try:
        triplets, node = extract_triplets(node)
        # return triplets
    # except Exception as e:
    except (exceptions.ServiceUnavailable, exceptions.TransientError) as e:
        # print("->", e)
        time.sleep(10)
        triplets, node = extract_triplets(node)
        # return triplets
    except (
        google.generativeai.types.generation_types.StopCandidateException,
        google.generativeai.types.generation_types.BlockedPromptException,
    ) as e:
        # print("-->", e)
        nodes = node.split()
        triplets = []
        for n in nodes:
            trplt = process_node(n)
            triplets.extend(trplt)

    return triplets


def split(node):
    global df
    global Settings
    start = node.metadata.get("start")
    end = node.metadata.get("end")

    mid = start + (end - start) // 2
    nodes = []

    left_text = " ".join(df.iloc[start:mid, 2].tolist()).lower().strip()
    right_text = " ".join(df.iloc[mid:end, 2].tolist()).lower().strip()

    if len(left_text) > 0:

        left_metadata = node.metadata.copy()
        left_metadata.update(
            {
                "block_size": Settings.chunk_size,
                "size": len(left_text) + 1,
                "start": start,
                "end": mid,
            }
        )
        left_node = Document(text=left_text, metadata=left_metadata)
        nodes.append(left_node)

    if len(right_text) > 0:

        right_metadata = node.metadata.copy()
        right_metadata.update(
            {
                "block_size": Settings.chunk_size,
                "size": len(right_text) + 1,
                "start": mid + 1,
                "end": end,
            }
        )
        right_node = Document(text=right_text, metadata=right_metadata)
        nodes.append(right_node)
    gc.collect()
    return nodes


setattr(Document, "split", split)


def triplet_extractor(text, metadata):

    unprocessed = []
    triplets_list = []
    doc = Document(text=text, metadata=metadata)
    try:
        triplets = process_node(doc)
        tag = f'{metadata["source"]}__{metadata["start"]}__{metadata["end"]}'
        triplets_list.append({"triplets": triplets, "id": tag})
    except google.generativeai.types.generation_types.BlockedPromptException as e:
        print(f"FAIL BlockedPromptException for {text}")
        print(e)
        unprocessed.append(metadata)
        triplets = []
    except google.generativeai.types.generation_types.StopCandidateException as e:
        print(f"FAIL StopCandidateException for {text}")
        print(e)
        unprocessed.append(metadata)
        triplets = []
    except (exceptions.ServiceUnavailable, exceptions.TransientError) as e:
        time.sleep(10)
        # print(e)
        triplets = process_node(doc)
        tag = f'{metadata["source"]}__{metadata["start"]}__{metadata["end"]}'
        triplets_list.append({"triplets": triplets, "id": tag})
    except Exception as e:
        print(f"FAIL Exception for {text}, \n{e}")
        unprocessed.append(metadata)
        print(e)
        time.sleep(3)
        triplets = []

    with open(TRIPLET_FILE, "a") as fd:
        for t in triplets_list:
            json.dump(t, fd)
            fd.write(",\n")

    with open(UNPROCESSED_FILE, "a") as fd:
        for t in unprocessed:
            json.dump(t, fd)
            fd.write(",\n")

    pbar.update(1)

    # del triplets_list
    # del unprocessed
    gc.collect()

    return triplets


# docs = docs[1728:1730]
with tqdm(total=len(docs)) as pbar:

    kg_index = KnowledgeGraphIndex.from_documents(
        docs,
        storage_context=storage_context,
        max_triplets_per_chunk=MAX_TRIPLETS,
        space_name=SPACE_NAME,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        kg_triplet_extract_fn=triplet_extractor,
        include_embeddings=True,
        verbose=True,
        timeout=100,
    )

with open(TRIPLET_FILE, "a") as fd:
    fd.write("]")

with open(UNPROCESSED_FILE, "a") as fd:
    fd.write("]")

kg_index.storage_context.persist(persist_dir=STORAGE_PATH)


# kg_indexes = load_indices_from_storage(storage_context=storage_context)


parameters = [
    (True, True, True),
    (False, True, False),
    (True, True, False),
    (False, False, False),
]


for param in parameters:
    INCLUDE_TEXT, VERBOSE, GLOBAL = param
    strategy_query_engines = {}

    strategy_query_engines["vector_based"] = kg_index.as_query_engine()

    strategy_query_engines["keyword-based"] = kg_index.as_query_engine(
        retriever_mode="keyword",
        response_mode="tree_summarize",
        verbose=VERBOSE,
        include_text=INCLUDE_TEXT,
        explore_global_knowledge=GLOBAL,
    )

    strategy_query_engines["hybrid"] = kg_index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
        verbose=VERBOSE,
        explore_global_knowledge=GLOBAL,
    )

    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        synonym_expand_policy="union",
        max_synonyms=5,
        retriever_mode="semantic",
        verbose=VERBOSE,
        include_text=INCLUDE_TEXT,
        explore_global_knowledge=GLOBAL,
    )
    strategy_query_engines["rag"] = RetrieverQueryEngine.from_args(graph_rag_retriever)

    strategy_query_engines["graph"] = KnowledgeGraphQueryEngine(
        storage_context=storage_context,
        llm=llm,
        refresh_schema=True,
        verbose=VERBOSE,
        include_text=INCLUDE_TEXT,
        explore_global_knowledge=GLOBAL,
    )
    strategy_query_engines["chain"] = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=cypher_model,
        qa_llm=qa_model,
        validate_cypher=True,
        return_intermediate_steps=INCLUDE_TEXT,
        verbose=VERBOSE,
    )

    questions = [
        "Tell me about the relationship between Vitamind D and Covid?",
        "Qual é a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar?",
        "Descreva a relação direta ou indireta entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar trazendo referências bibliográficas conhecidas que suportem a resposta.",
        "O que é deficiência de vitamina D?",
        "O que é vitamina D e como ela pode interferir na recuperação da COVID-19?",
        "Quem é Silvio Santos?",
        "what is vitamin d insufficiency?",
        "what is the ideal vitami D serum concentration for a human being?",
        "what is vitamin d deficiency?",
        "What is the relationship between the COVID-19 infection, Vitamin D insufficiency and deficiency?",
        "What is the relationship between the severity, recovery, and COVID-19 infection, Vitamin D insufficiency and deficiency, and the use of sunscreen?",
        "Trace the direct or indirect relationship between the severity, recovery, and COVID-19 infection, Vitamin D insufficiency and deficiency, and the use of sunscreen, bringing known bibliographic references that support the answer.",
        "Who is Silvio Santos?",
        "¿Qué es la insuficiencia de vitamina D?",
        "¿Cuál es la concentración ideal de vitamina D en suero para un ser humano?",
        "¿Qué es la deficiencia de vitamina D?",
        "¿Cuál es la relación entre la infección por COVID-19, la insuficiencia y la deficiencia de vitamina D?",
    ]

    answers_map = []
    t1 = time.time()

    for stg_name, stg in strategy_query_engines.items():

        q_id = 0

        while q_id < len(questions):
            question = questions[q_id]
            try:
                tic = time.time()
                response = stg.query(question)
                tac = time.time() - tic
                fact_score = evaluator.evaluate_response(response=response)
                answer = {
                    "db": DB_ID,
                    "strategy": stg_name,
                    "question": question,
                    "answer": response.response,
                    "answer_context": response,
                    "eval": fact_score.score,
                    "eval_context": "\n".join(fact_score.contexts),
                    "time": tac,
                }
                answers_map.append(answer)
                q_id += 1
                # print(answer)
            except Exception as e:
                time.sleep(random.randint(5, 25))
                # print(DB_ID, stg_name)

pd.DataFrame(answers_map).to_csv(
    f"v_qa_global_{GLOBAL}_verbose_{VERBOSE}_{DB_ID}_include_text_{INCLUDE_TEXT}_{CHUNK_SIZE}_{MAX_TRIPLETS}_overlap_total_en.csv",
    index=None,
)
print("elapsed: ", time.time() - t1)
