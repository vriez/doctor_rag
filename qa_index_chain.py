import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.schema import Node, Document
from llama_index.core import load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
    StorageContext,
    ServiceContext,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.indices.loading import load_indices_from_storage
from llama_index.core.evaluation import FaithfulnessEvaluator

llm = Gemini(temperature=0, timeout=60)
embedding_llm = GeminiEmbedding(model="models/embedding-001")
evaluator = FaithfulnessEvaluator(llm=llm)

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 512

INCLUDE_TEXT = True
VERBOSE = False
GLOBAL = True

database = "neo4j"
username = "neo4j"

# Credentials loaded from environment variables.
# Set NEO4J_AUTH_MAP as a JSON string, e.g.:
# export NEO4J_AUTH_MAP='{"db_id": {"username": "neo4j", "password": "...", "url": "bolt://..."}}'
auth_map_json = os.environ.get("NEO4J_AUTH_MAP")
if not auth_map_json:
    raise EnvironmentError(
        "NEO4J_AUTH_MAP environment variable is required. "
        "Set it as a JSON string mapping database IDs to credentials: "
        '{"db_id": {"username": "...", "password": "...", "url": "bolt://..."}}'
    )
auth_map = json.loads(auth_map_json)


for database, (creds) in auth_map.items():

    username, password, url = creds.values()
    print(database, username, url)
    storage_path = f"./storage_graph_{database}__2048"
    storage_path = f"./storage_graph_{database}_overlap_286__2048"

    graph_store = Neo4jGraphStore(
        username=username, password=password, url=url, database="neo4j", timeout=60
    )
    storage_context = StorageContext.from_defaults(
        graph_store=graph_store,
        persist_dir=storage_path,
    )
    auth_map[database]["storage"] = storage_context
    index = load_indices_from_storage(storage_context=storage_context)[0]

    auth_map[database]["index"] = index
    auth_map[database]["strategy"] = {}

    parameters = [
        (True, True, True),
        (False, True, False),
        (True, True, False),
        (False, False, False),
    ]

    for i, param in enumerate(parameters):

        auth_map[database]["strategy"][f"verbo_based__{i}"] = index.as_query_engine()

        auth_map[database]["strategy"][f"keyword_based__{i}"] = index.as_query_engine(
            retriever_mode="keyword",
            response_mode="tree_summarize",
            verbose=VERBOSE,
            include_text=INCLUDE_TEXT,
            explore_global_knowledge=GLOBAL,
        )

        auth_map[database]["strategy"][f"hybrid__{i}"] = index.as_query_engine(
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

        auth_map[database]["strategy"][f"rag__{i}"] = RetrieverQueryEngine.from_args(
            graph_rag_retriever
        )

        auth_map[database]["strategy"][f"kg__{i}"] = KnowledgeGraphQueryEngine(
            storage_context=storage_context,
            llm=llm,
            refresh_schema=True,
            verbose=VERBOSE,
            include_text=INCLUDE_TEXT,
            explore_global_knowledge=GLOBAL,
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
for db, exp in auth_map.items():
    for stg_name, stg in exp.get("strategy", {}).items():

        for question in questions:

            try:
                tic = time.time()
                response = stg.query(question)
                tac = time.time() - tic
                fact_score = evaluator.evaluate_response(response=response)
                answer = {
                    "db": db,
                    "strategy": stg_name,
                    "question": question,
                    "answer": response.response,
                    "answer_context": response,
                    "eval": fact_score.score,
                    "eval_context": "\n".join(fact_score.contexts),
                    "time": tac,
                }
                answers_map.append(answer)
            except Exception as e:
                print(db, stg_name, e)
                time.sleep(1)

print("elapsed: ", time.time() - t1)

pd.DataFrame(answers_map).to_csv(
    f"QA_global_all_total_en.csv",
    index=None,
)
