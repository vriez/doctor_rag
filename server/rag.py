import re
import json
import neo4j
import pydantic

from langchain.embeddings import (
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Optional, Dict, Any
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain.chains import GraphCypherQAChain
from langchain.pydantic_v1 import Field, BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.schema.output_parser import StrOutputParser

from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import create_extraction_chain_pydantic
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter


# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="mistral")

neo4j_url = "bolt://localhost:7687"
neo4j_username = "neo4j"
neo4j_password = "password"
graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        # print("ingesting: ", pdf_file_path)
        # docs = PyPDFLoader(file_path=pdf_file_path).load()
        # chunks = self.text_splitter.split_documents(docs)
        # chunks = filter_complex_metadata(chunks)

        # vector_store = Neo4jVector.from_documents(
        #     # chunks,
        #     docs,
        #     embeddings,
        #     url=neo4j_url,
        #     username=neo4j_username,
        #     password=neo4j_password,
        # )

        # self.retriever = vector_store.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"k": 3, "score_threshold": 0.5},
        # )
        # # json_parser = SimpleJsonOutputParser()

        # self.retriever = vector_store.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={
        #         "k": 3,
        #         "score_threshold": 0.5,
        #     },
        # )

        # self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
        #               | self.prompt
        #               | self.model
        #               | StrOutputParser())
        qa_model = ChatOllama(model="mistral", temperature=0.0)
        cypher_model = ChatOpenAI(model="gpt-4", temperature=0.0)

        self.chain = GraphCypherQAChain.from_llm(
            graph=graph,
            cypher_llm=cypher_model,
            qa_llm=qa_model,
            validate_cypher=True,
            verbose=True,
        )

    def ask(self, query: str):
        # if not self.chain:
        #     return "Please, add a PDF document first."
        print("chain: ", self.chain)
        qa = self.chain.invoke(query)
        return qa.get("result", "NOT FOUND")

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
