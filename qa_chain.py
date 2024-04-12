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

# qa_model = ChatOllama(model="mistral", temperature=0.0)
# qa_model = ChatOllama(model="mistral", temperature=0.0)

# cypher_model = ChatVertexAI(
#     model_name="gemini-pro", temperature=0.0, convert_system_message_to_human=True
# )
# qa_model = ChatVertexAI(
#     model_name="gemini-pro", temperature=0.0, convert_system_message_to_human=True
# )

# cypher_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# qa_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

auth_map = {
    "aa71f7f54748577d4ac173a4462cd074": {
        "username": "neo4j",
        "password": "nail-interface-necks",
        "url": "bolt://3.238.101.93:7687",
    },
    "26d1b177537db8832f0d69488ed8fa41": {
        "username": "neo4j",
        "password": "machines-electrolytes-troop",
        "url": "bolt://3.239.198.148:7687",
    },
    "68db8edf1c2e12a3cc7f7860fe28d770": {
        "username": "neo4j",
        "password": "bureau-auto-adherences",
        "url": "bolt://100.27.45.13:7687",
    },
    "bcd95aaad62b7b424b7f0675feac7185": {
        "username": "neo4j",
        "password": "accusation-tube-blueprints",
        "url": "bolt://3.91.206.92:7687",
    },
}

for database, (creds) in auth_map.items():

    username, password, url = creds.values()
    print(database, username, password, url)
    storage_path = f"./storage_graph_{database}__2048"

    graph = Neo4jGraph(url=url, username=username, password=password, database="neo4j")
    # auth_map[database]["graph"] = graph
    auth_map[database]["chain"] = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=cypher_model,
        qa_llm=qa_model,
        validate_cypher=True,
        # return_intermediate_steps=True,
        verbose=True,
    )

queries = [
    "Qual é a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar?",
    "Trace a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar trazendo referências bibliográficas conhecidas que suportem a resposta.",
    "Quem é Silvio Santos?",
    # "O que é vitamina D?",
    # "what is Vitamin D",
    # "O que é COVID-19?",
    # "Quais são os sinônimos da Vitamina D?",
]

data = []
for db, conn in auth_map.items():
    # print("chain: ", chain)
    chain = conn["chain"]
    for question in queries:
        # try:
        tic = time.time()
        answer = chain.invoke(question)
        # print(answer)
        d = {
            "db": db,
            "question": question,
            "answer": answer,
            "time": time.time() - tic,
        }
        data.append(d)
        # except Exception as e:
        #     print(f"--{e}--")

pd.DataFrame(data).to_csv(
    f"QA_openai_gemini.csv",
    index=None,
)
