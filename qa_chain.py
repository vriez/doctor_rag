from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

# from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

# neo4j_url = "bolt://localhost:7687"
# neo4j_username = "neo4j"
# neo4j_password = "password"

# username = "neo4j"
# chunk_size = 512
# password = "mints-indication-topic"
# url = "bolt://REDACTED_IP:7687"
# # bolt+s://d1d75140b2b1d7fd08d143a30d6c2730.neo4jsandbox.com:7687

# 1536
# username = "neo4j"
# chunk_size = 1536
# password = "letterhead-butters-clips"
# url = "bolt://REDACTED_IP:7687"
# bolt+s://c5ddd8a8c31c239cc0ba42fe96f5bb17.neo4jsandbox.com:7687

# 2048
# username = "neo4j"
# chunk_size = 2048
# password = "procurement-pine-henrys"
# url = "bolt://REDACTED_IP:7687"
# bolt+s://81b1dfa2b516e98f8fb2f91269bf7416.neo4jsandbox.com:7687

username = "neo4j"
password = "barriers-brush-stocking"
url = "bolt://REDACTED_IP:7687"

graph = Neo4jGraph(url=url, username=username, password=password, database="neo4j")

cypher_model = ChatOpenAI(model="gpt-4", temperature=0.0)
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

cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=cypher_model,
    qa_llm=qa_model,
    validate_cypher=True,
    return_intermediate_steps=True,
    verbose=True,
)

queries = [
    "Qual é a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar?",
    "Trace a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar trazendo referências bibliográficas conhecidas que suportem a resposta.",
    "Quem é Silvio Santos?",
    "O que é vitamina D?",
    "what is Vitamin D",
    "O que é COVID-19?",
    "Quais são os sinônimos da Vitamina D?",
]

for question in queries:
    try:
        answer = cypher_chain.invoke(question)
        print(answer)
    except Exception as e:
        print(f"--{e}--")
