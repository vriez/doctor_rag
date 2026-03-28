import os
import time
import pandas as pd
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

# from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

cypher_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
qa_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Credentials loaded from environment variables.
# Set NEO4J_AUTH_MAP as a JSON string, e.g.:
# export NEO4J_AUTH_MAP='{"db_id": {"username": "neo4j", "password": "...", "url": "bolt://..."}}'
import json

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

    graph = Neo4jGraph(url=url, username=username, password=password, database="neo4j")
    auth_map[database]["chain"] = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=cypher_model,
        qa_llm=qa_model,
        validate_cypher=True,
        verbose=True,
    )

queries = [
    "Qual é a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar?",
    "Trace a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar trazendo referências bibliográficas conhecidas que suportem a resposta.",
    "Quem é Silvio Santos?",
]

data = []
for db, conn in auth_map.items():
    chain = conn["chain"]
    for question in queries:
        tic = time.time()
        answer = chain.invoke(question)
        d = {
            "db": db,
            "question": question,
            "answer": answer,
            "time": time.time() - tic,
        }
        data.append(d)

pd.DataFrame(data).to_csv(
    f"QA_openai_gemini.csv",
    index=None,
)
