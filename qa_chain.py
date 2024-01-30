from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOllama, ChatOpenAI

neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "password"

graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)

cypher_model = ChatOpenAI(model="gpt-4", temperature=0.0)
qa_model = ChatOllama(model="mistral", temperature=0.0)
# qa_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=cypher_model,
    qa_llm=qa_model,
    validate_cypher=True,
    verbose=True,
)

query_1 = "Qual é a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar?"
query_2 = "Trace a relação entre a severidade, recuperação e infecção à COVID 19, insuficiência e deficiência de Vitamina D e o uso de protetor solar trazendo referências bibliográficas conhecidas que suportem a resposta."
query_3 = "Quem é Silvio Santos?"

for question in [query_1, query_2, query_3]:
    answer = cypher_chain.invoke(question)
    print(answer)