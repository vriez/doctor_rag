import re
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# from pydash import snake_case
from itertools import product
from langchain.embeddings import (
    OllamaEmbeddings,
    # SentenceTransformerEmbeddings,
)
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from typing import List, Optional

# from langchain.chains import GraphCypherQAChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import (
    PydanticOutputParser,
    PandasDataFrameOutputParser,
    YamlOutputParser,
    CommaSeparatedListOutputParser,
    # ListOutputParser,
)

# from langchain.chains import create_extraction_chain_pydantic
# from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    # BibtexLoader,
    # WikipediaLoader,
    PubMedLoader,
    WebBaseLoader,
)
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.pydantic_v1 import Field, BaseModel, ConstrainedList
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain.pydantic_v1 import Field, BaseModel

# embeddings = OllamaEmbeddings(model="mistral")


cols = ["subject_name", "subject_type", "relationship", "object_name", "object_type"]

# CHUNK_SIZE = 1024
# CHUNK_OVERLAP = 24
# CHUNK_SIZE = 2048*2
# CHUNK_OVERLAP = 0

# neo4j_url = "bolt://localhost:7687"
# neo4j_username = "neo4j"
# neo4j_password = "password"
# graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)

model = ChatOllama(model="mistral", temperature=0.0)


# Set up a parser + inject instructions into the prompt template.
def transform_cell(cell_value):
    # print("cell_value: ", cell_value)
    cell_value = " ".join(
        re.sub(r"(?<=[a-z])(?=[A-Z](?![A-Z]))", " ", cell_value).lower().split()
    )
    # cell_value = snake_case(cell_value).replace("_", " ")
    return cell_value


def parse(data):
    text = "".join(data)
    pattern = r"- (\d+)\. - (.*?) <@>"

    matches = re.findall(pattern, text, re.DOTALL)

    result_list = [(match[0], match[1].strip()) for match in matches]

    df_data = [d.split("_-.-_") for _, d in result_list]
    df_data = [d for d in df_data if len(d) == 5]

    result = pd.DataFrame(df_data, columns=cols)
    result = result.applymap(transform_cell)
    return result


class Property(BaseModel):
    """A single property consisting of key and value"""

    key: str = Field(..., description="key")
    value: str = Field(..., description="value")


class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties"
    )


class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )


class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""

    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )


parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

prompt_message = """
Domain:
    - biology
    - biochemistry
    - endocrinology
    - dermatology
    - medical sciences
    - clinical trials
    - virology
    - nephrology
    - gastroenterology
    - hepatology
    - immunology
    - osteology
    - posology
    - statistics

Task:
    - Consider Domain: Consider the context given to accurately determine triples.
    - Extract key Concepts and Relationships: Identify key concepts and relationships in the given input text.
    - Prioritize Salience: Focus on extracting the most important and meaningful concepts and relationships.
    # - Explore Alternative Interpretations: If possible, generate multiple interpretations of the input text to capture different perspectives and potential relationships.
    - Generate triples: A triple, `subject-predicate-object`, that represents the relationship between two concepts.   
    - Triple expansion: If any triple component contains a conjunction, expand it.

# Labeling Concepts:
#     - **Consistency**: Ensure you use basic or elementary types for concepts labels.
#     - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
#     - **Concepts Names**: Concepts name should be names or human-readable identifiers found in the text.
#     - **Concepts Types**: Concepts types should be domain-specific identifiers found in the text.
#     - **Semantics**: Concepts must not contain conjunctions, prepositions, interjection, pronouns, participle nor gerund.

# Labeling Relationships:
#     - **Consistency**: Ensure you use basic or elementary types for relationships labels.
#     - **Relationships**: Concepts types should be domain-specific identifiers found in the text.
#     - **Semantics**: Relationships must not contain conjunctions, prepositions, interjection, pronouns, participle nor gerund.
#     - **Case**: Relationships must be lowercase and separated by spaces
#     - **Concise**: Relationships must be as meaningful and compact as possible. It must not contain more than 3 words

# Handling Numerical Data and Dates:
# - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
# - **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
# - **Quotation Marks**: Never use escaped single or double quotes within concept's attributes values.
# - **Naming Convention**: Use plain text for concepts names, type as well as for relationships.
# - **Posology**: Are formed by concepts (patient, dosage, 20 mg/L) and relationships (ingest, take).
# - **Clinical Trials**: Attain the trial's hypothesis, metrics and results. So as to be allow argumentation and comparison between multiple clinical trials

Format:
    - Structured Output: The output must be a valid csv 5-elements object, where each triple contains the following list of attributes: subject_name", "subject_type", "relationship", "object_name", "object_type"; in this order.
    
Example:

    Input [plain text]: "Albert Einstein is best known for developing the theory of relativity, which revolutionized physics."
   
    Output [list of triples]:

        - 1. - `Albert Einstein`_-.-_`person`_-.-_`developed`_-.-_`Theory of Relativity`_-.-_`theory` <@>
        - 2. - `Theory of Relativity`_-.-_`theory`_-.-_`revolutionized`_-.-_`Physics`_-.-_`field of study` <@>


Additional Considerations:
    # - provide as many triples as you judge meaningful for the given input
    - each triple must contain 5 elements
    - each sub-sentence can have up to 7 triples
    - revisit the triples with more that 5 elements, if not comply remove it
    - each triple object must be a valid csv object
    - the output must be an enumerated list of triples
    - concepts can not be a conjunction, null, None or empty string
    - if a concept or relationship contain a conjunction, expand the whole triple into two conjunction-free triples
    - relationships can not be a conjunction, null, None or empty string
    # - the output must not contain any comments
    # - the output must not contain any non csv data
    # - the output must not contain any programming language
    # - the output must start with '- 1.' and must end with <@>

Format Instructions: {{ format_instructions }}
Input: {{ unstructured_text }}
"""


prompt = PromptTemplate(
    template=prompt_message,
    input_variables=["unstructured_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    format="j",
    template_format="jinja2",
    validate_template=True,
)


chain = {"unstructured_text": RunnablePassthrough()} | prompt | model | parser

CHUNK_SIZE = [512, 1024, 2048, 4096]
CHUNK_OVERLAP = [0, 24, 56]

loader = PyPDFDirectoryLoader("dataset")
docs = loader.load()

output_folder = Path("py_pdf_directory_loader")
counter = 0
fname = None
dok = None
data = ""
for doc in docs:
    fname = Path(doc.metadata["source"])
    if Path(output_folder / f"{fname.stem}.txt").exists():
        continue
    if not bool(dok):
        dok = fname
    if dok != fname:
        dok = fname
        counter = 0
        with open(output_folder / f"{fname.stem}.txt", "w") as f:
            f.write(data)
        data = ""

    # print(f"{fname.stem}_{counter}.txt")
    counter += 1
    data += "\n" + data

from pypdf import PdfReader

# for chunk_size, chunk_overlap in product(CHUNK_SIZE, CHUNK_OVERLAP):
#     output_dir = Path(f"cleaned_data__{chunk_size}_{chunk_overlap}")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     graph_dir = output_dir / "graph"
#     graph_dir.mkdir(parents=True, exist_ok=True)

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     chunks = text_splitter.split_documents(docs)

#     offset = 0
#     chunks = chunks[offset:]

#     for i, chunk in tqdm(enumerate(chunks, offset), total=len(chunks)):

#         file_name = Path(chunk.metadata.get("source")).stem
#         doc_text = chunk.page_content.replace("-\n", "")
#         # doc_text = re.sub(r'(?<!\.)\n', ' ', doc_text)

#         if not Path(graph_dir / f"{i}_{file_name}.csv").exists():
#             data = chain.invoke({"unstructured_text": doc_text})
#             f = Path(chunk.metadata.get("source")).stem
#             df = parse(data)
#             df.to_csv(graph_dir / f"{i}_{file_name}.csv", index=None)

#         gc.collect()
#     gc.collect()

#     # nodes = []
#     # rels = []
#     # for trp in data.triples:
#     #     subject_node = Node(id=trp.subject_name, type=trp.subject_type)
#     #     object_node = Node(id=trp.object_name, type=trp.object_type)
#     #     rel = Relationship(
#     #         source=subject_node,
#     #         target=object_node,
#     #         type=trp.relationship
#     #     )
#     #     nodes.extend([subject_node, object_node])
#     #     rels.append(rel)

#     # # Print or use the resulting GraphDocument
#     # graph_document = GraphDocument(
#     #     nodes=nodes,
#     #     relationships=rels,
#     #     source=chunk,
#     # )
#     # # print(graph_document)

#     # graph.add_graph_documents([graph_document])

#     # cypher_chain = GraphCypherQAChain.from_llm(
#     #     graph=graph,
#     #     cypher_llm=model,
#     #     qa_llm=model,
#     #     validate_cypher=True,
#     #     verbose=True,
#     # )

#     # def ask(self, query: str):
#     #     if not chain:
#     #         return "Please, add a PDF document first."
#     #     graph_result = cypher_chain.run(query)
#     #     # graph_result = chain.invoke(query)
#     #     return graph_result

#     # def clear(self):
#     #     vector_store = None
#     #     retriever = None
#     #     chain = None
