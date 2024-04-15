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
username = "neo4j"

# 2048 incremental
# password = "nail-interface-necks"
# url = "bolt://3.238.101.93:7687"
# db_id = "aa71f7f54748577d4ac173a4462cd074"

# username = "neo4j"
# password = "machines-electrolytes-troop"
# url = "bolt://3.239.198.148:7687"
# db_id = "26d1b177537db8832f0d69488ed8fa41"

# username = "neo4j"
# password = "bureau-auto-adherences"
# url = "bolt://100.27.45.13:7687"
# db_id = "68db8edf1c2e12a3cc7f7860fe28d770"

# username = "neo4j"
# password = "accusation-tube-blueprints"
# url = "bolt://3.91.206.92:7687"
# db_id = "bcd95aaad62b7b424b7f0675feac7185"


# auth_map = {
#     "aa71f7f54748577d4ac173a4462cd074": {
#         "username": "neo4j",
#         "password": "nail-interface-necks",
#         "url": "bolt://3.238.101.93:7687",
#     },
#     "26d1b177537db8832f0d69488ed8fa41": {
#         "username": "neo4j",
#         "password": "machines-electrolytes-troop",
#         "url": "bolt://3.239.198.148:7687",
#     },
#     "68db8edf1c2e12a3cc7f7860fe28d770": {
#         "username": "neo4j",
#         "password": "bureau-auto-adherences",
#         "url": "bolt://100.27.45.13:7687",
#     },
#     "bcd95aaad62b7b424b7f0675feac7185": {
#         "username": "neo4j",
#         "password": "accusation-tube-blueprints",
#         "url": "bolt://3.91.206.92:7687",
#     },
# }

auth_map = {
    "d29e690203979119220cf60f40490e26": {
        "username": "neo4j",
        "password": "samples-rush-capability",
        "url": "bolt://34.234.207.173:7687",
    },
    "7effdf391d1301b4dff0fdd97838e307": {
        "username": "neo4j",
        "password": "thresholds-checker-science",
        "url": "bolt://35.173.122.197:7687",
    },
    "69687aead0421f367f69aac6f1249cb2": {
        "username": "neo4j",
        "password": "irons-hiss-preventions",
        "url": "bolt://54.227.0.198:7687",
    },
    "8d02074acea413208355e0c16b66dc4e": {
        "username": "neo4j",
        "password": "stump-rain-ladders",
        "url": "bolt://3.216.133.253:7687",
    },
}


for database, (creds) in auth_map.items():

    username, password, url = creds.values()
    print(database, username, password, url)
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
    f"QA_global_all_total_en.csv",
    index=None,
)
