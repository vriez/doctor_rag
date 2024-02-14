
import os
import logging
import sys
import pandas as pd
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  
# from langchain.docstore.document import Document
from llama_index import Document
from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.query_engine import KnowledgeGraphQueryEngine

from llama_index.graph_stores import NebulaGraphStore
from llama_index.graph_stores import Neo4jGraphStore

from llama_index.llms import Ollama, OpenAI

from llama_index.embeddings import OllamaEmbedding


# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-16k")
#llm = Ollama(temperature=0, model="mistral")

llm = Ollama(model="mistral", temperature=0.0, context_window=4900, request_timeout=300)
embedding_llm = OllamaEmbedding(model_name="mistral")

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_llm)

from llama_index import SimpleDirectoryReader

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"


# reader = SimpleDirectoryReader(input_dir="../data/knowledge graphs/rebel_llamaindex/wiki/")
# documents = reader.load_data()

df = pd.read_csv("doc_clean_data.csv")

docs = df.groupby("0")
documents = []
for fname, doc in docs:
    page_count = 0
    block = ""
    for i, row in doc.iterrows():
        content = row["1"]
        # print(fname, i, len(content))

        if len(block) + len(content) <= 4000:
            # page_count += len(content)
            block += (" " + content)
            page_content = block
        else: 
            # print("page_content: ", fname, len(page_content))
            # page_count = 0
            block = content
        d = Document(
            text = page_content,
            # source=fname,
            # get_doc_id=fname,
        )
        documents.append(d)
    page_content = " ".join(doc["1"])
    # print(fname, len(page_content))


space_name = "covid_relationships"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  
tags = ["entity"]

# graph_store = NebulaGraphStore(
#     space_name=space_name,
#     edge_types=edge_types,
#     rel_prop_names=rel_prop_names,
#     tags=tags,
# )


username = "neo4j"
password = "9eGGLTE9W13bttviLjMmQRpzaxYzWSmhiW64CrGi40w"
url = "neo4j+s://4feea599.databases.neo4j.io"
database = "neo4j"

# username = "neo4j"
# password = "password"
# url = "neo4j://127.0.0.1:7687"
# database = "neo4j"

graph_store = Neo4jGraphStore(
    username=username, password=password, url=url, database=database
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)


kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=5,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)


query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)