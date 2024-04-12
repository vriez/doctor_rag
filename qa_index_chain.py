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
# llm = OpenAI(temperature=0.0, model="gpt-4")
# embedding_llm = OpenAIEmbedding(model="text-embedding-3-small")
evaluator = FaithfulnessEvaluator(llm=llm)

Settings.llm = llm
Settings.embed_model = embedding_llm
Settings.chunk_size = 512

INCLUDE_TEXT = True
VERBOSE = False
GLOBAL = True
# space_name = "llamaindex"
# edge_types, rel_prop_names = ["relationship"], ["property"]
# tags = ["entity"]

database = "neo4j"

# 2048 incremental
username = "neo4j"
password = "REDACTED_NEO4J_PASSWORD"
url = "bolt://REDACTED_IP:7687"
db_id = "aa71f7f54748577d4ac173a4462cd074"

username = "neo4j"
password = "REDACTED_NEO4J_PASSWORD"
url = "bolt://REDACTED_IP:7687"
db_id = "26d1b177537db8832f0d69488ed8fa41"

username = "neo4j"
password = "REDACTED_NEO4J_PASSWORD"
url = "bolt://REDACTED_IP:7687"
db_id = "68db8edf1c2e12a3cc7f7860fe28d770"

username = "neo4j"
password = "REDACTED_NEO4J_PASSWORD"
url = "bolt://REDACTED_IP:7687"
db_id = "bcd95aaad62b7b424b7f0675feac7185"


auth_map = {
    "aa71f7f54748577d4ac173a4462cd074": {
        "username": "neo4j",
        "password": "REDACTED_NEO4J_PASSWORD",
        "url": "bolt://REDACTED_IP:7687",
    },
    "26d1b177537db8832f0d69488ed8fa41": {
        "username": "neo4j",
        "password": "REDACTED_NEO4J_PASSWORD",
        "url": "bolt://REDACTED_IP:7687",
    },
    "68db8edf1c2e12a3cc7f7860fe28d770": {
        "username": "neo4j",
        "password": "REDACTED_NEO4J_PASSWORD",
        "url": "bolt://REDACTED_IP:7687",
    },
    "bcd95aaad62b7b424b7f0675feac7185": {
        "username": "neo4j",
        "password": "REDACTED_NEO4J_PASSWORD",
        "url": "bolt://REDACTED_IP:7687",
    },
}

for database, (creds) in auth_map.items():

    username, password, url = creds.values()
    print(database, username, password, url)
    storage_path = f"./storage_graph_{database}__2048"

    graph_store = Neo4jGraphStore(
        username=username, password=password, url=url, database="neo4j", timeout=60
    )
    storage_context = StorageContext.from_defaults(
        graph_store=graph_store,
        persist_dir=storage_path,
    )
    auth_map[database]["storage"] = storage_context
    # try:
    # index = load_index_from_storage(storage_context=storage_context)
    # try:
    index = load_indices_from_storage(storage_context=storage_context)[0]
    # except:
    #     index = load_indices_from_storage(storage_context=storage_context)[0]

    # except Exception as e:
    #     print(e)
    #     continue
    auth_map[database]["index"] = index

    auth_map[database]["strategy"] = {}
    # auth_map[database]["strategy"]["vector_based"] = index.as_query_engine(
    #     explore_global_knowledge=GLOBAL,
    #     verbose=VERBOSE,
    #     include_text=INCLUDE_TEXT,
    # )

    auth_map[database]["strategy"]["keyword-based"] = index.as_query_engine(
        include_text=INCLUDE_TEXT,
        retriever_mode="keyword",
        response_mode="tree_summarize",
        # explore_global_knowledge=GLOBAL,
        verbose=VERBOSE,
    )

    auth_map[database]["strategy"]["hybrid"] = index.as_query_engine(
        include_text=INCLUDE_TEXT,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=3,
        # explore_global_knowledge=GLOBAL,
        verbose=VERBOSE,
    )

    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        synonym_expand_policy="union",
        max_synonyms=7,
        retriever_mode="semantic",
        with_nl2graphquery=True,
        verbose=VERBOSE,
        include_text=INCLUDE_TEXT,
    )

    auth_map[database]["strategy"]["rag"] = RetrieverQueryEngine.from_args(
        graph_rag_retriever
    )

    kg_retriever = index.as_retriever(include_text=INCLUDE_TEXT, verbose=VERBOSE)
    auth_map[database]["strategy"]["kg"] = RetrieverQueryEngine.from_args(kg_retriever)

questions = [
    # "Tell me about the relationship between Vitamind D and Covid?",
    # "Tell me about the relationship between Vitamind D and Covid?",
    # "O que é deficiência de vitamina D?",
    # "o que é insuficiencia de vitamina D?",
    "what is vitamin d insufficiency?",
    "what is vitamin d deficiency?",
    # "o que é insuficiencia de vitamina D?",
    # "O que é vitamina D e como ela pode interferir na recuperação da COVID-19?",
    # "Qual é a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar?",
    # "Trace a relação direta ou indireta entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar trazendo referências bibliográficas conhecidas que suportem a resposta.",
    # "Quem é Silvio Santos?",
    "What is the relationship between the COVID-19 infection, Vitamin D insufficiency and deficiency?"
    "What is the relationship between the severity, recovery, and COVID-19 infection, Vitamin D insufficiency and deficiency, and the use of sunscreen?",
    "Trace the direct or indirect relationship between the severity, recovery, and COVID-19 infection, Vitamin D insufficiency and deficiency, and the use of sunscreen, bringing known bibliographic references that support the answer.",
    "Who is Silvio Santos?",
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
                # print(stg_name, response.response)
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
                # answers_map[db][stg_name][question] = response.response
                # display(Markdown(f"<b>{response}</b>"))
            except Exception as e:
                print(db, stg_name, e)
                # continue
            time.sleep(1)
            # print()
            # print()
            # print(answers_map)

print("elapsed: ", time.time() - t1)

pd.DataFrame(answers_map).to_csv(
    f"QA_global_{GLOBAL}_verbose_{VERBOSE}_gemini_include_text_{INCLUDE_TEXT}_0_total_en.csv",
    index=None,
)
