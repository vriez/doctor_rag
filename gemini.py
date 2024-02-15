from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import (
    PydanticOutputParser,
    PandasDataFrameOutputParser,
    YamlOutputParser,
    CommaSeparatedListOutputParser,
    # ListOutputParser,
)
from langchain.pydantic_v1 import Field, BaseModel
from typing import List, Optional
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema.output_parser import StrOutputParser

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

llm = ChatGoogleGenerativeAI(model="gemini-pro")

prompt_msg = """
Goal: Extract information from text and construct a knowledge graph.

Nodes: Represent entities and concepts, similar to Wikipedia pages.

Knowledge Graph Structure:

    Properties: 
        Key-value pairs (camelCase keys) for attributes (e.g., "birthDate").
        Numerical data and dates integrated as properties.
        Allowed data types: Text, numbers, booleans, dates, lists.
        Enrichment: Include relevant information from the text and external sources (optional). Examples: definitions, related entities, disambiguations, confidence scores.
    Nodes:
        Type: Consistent basic types like "Person", "Place", "Event", etc.
        Additional types allowed for clarity (e.g., "Scientist" for a "Person").
        ID: Human-readable identifiers from the text (not integers).
        Allowed data types: Text, numbers, dates.
        Properties: Key-value pairs (camelCase keys) for attributes (e.g., "birthDate").
            Numerical data and dates integrated as properties.
            Consider separate nodes for significant historical/scientific dates/numbers.
    Relationships:
        Verbs describe connections between nodes (e.g., "developed", "located in").
        Properties can enrich relationships (e.g., "year" for "developed").
        Allowed data types: Text, numbers, dates.

Enrichment:
    Add semantic information from the properties so as to make enhance the meaning of the relationships and nodes.

Consistency:

    Ensure consistent entity representation throughout the graph (coreference resolution).
    Use the most complete identifier when referring to the same entity multiple times.

Error Handling:

    Retry on specific errors with clear retry logic and failure messages.
    Indicate retryable and non-retryable errors.

Flexibility:

    Allow user-defined node types and properties for specific domains.
    Consider user feedback to improve knowledge graph extraction.

Examples:

    Albert Einstein is best known for developing the theory of relativity

    yields

    "nodes": [
      {
        "id": "Albert Einstein",
        "type": "Person",
        "properties": [
          {
            "key": "role",
            "value": "Scientist"
          },
          {
            "key": "profession",
            "value": "Physicist"
          },
          {
            "key": "nationality",
            "value": "German"
          },
          {
            "key": "birth date",
            "value": "March 14, 1879"
          }
        ]
      },
      {
        "id": "Theory of Relativity",
        "type": "Theory",
        "properties": [
          {
            "key": "field of study",
            "value": "Physics"
          }
        ]
      }
    ],
    "rels": [
      {
        "source": {
          "id": "Albert Einstein",
          "type": "Physicist"
        },
        "target": {
          "id": "Theory of Relativity",
          "type": "Theory"
        },
        "type": "developed",
        "properties": [
            {
              "key": "year",
              "value": "1905"
            },
            {
              "key": "published at",
              "value": "On the Electrodynamics of Moving Bodies"
            }
          ]
      }
    ]
}


Output Formats:

    JSON

Remember:

    Adhere to the outlined rules for accurate knowledge graph construction.
    Strive for simplicity and clarity in the extracted knowledge.
    No nodes are 

Format Instructions: {{ format_instructions }}
Input: {{ unstructured_text }}

"""

extract_msg = """
take the input text and follow the procedure step by step bellow:
0. correct the punctuation, synthax and morphology of the input text.
1. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs:
2. organize 2.'s the schema below such that:
  - properties are objects with attributes key[string] and value[string]. neither key nor value can contain `,`
  - Poperty value is a compound noun, adjective or compound adjective and must not be infered
  - Property keys is the infered type of the property value in the biomedical context
  - nodes are objects with attributes id[string], type[string] and properties[list[Property]]
  - nodes are nouns whose properties are compounds nouns, adjectives and compound adjectives
  - Node id is noun or compound noun not a verb or compound adjective used as a noun
  - Node type is the infered type of the property value in the biomedical context
  - Node properties is a list properties that are linked to the Node`s id
  - relationships are objects with attributes source[Node], target[Node], type[string] and properties[list[Property]]
  - Relationship type is the verb that describes the interaction between source and target in the correct direction
  - Relationship properties are the present adverbs and compound adjectives as well as the infered meaninful properties
  - 

3. one example is given by:

  input:
  
      Albert Einstein is best known for developing the theory of relativity
  
  output:

    {
        "nodes": [
          {
            "id": "Albert Einstein",
            "type": "Person",
            "properties": [
              {
                "key": "role",
                "value": "Scientist"
              },
              {
                "key": "profession",
                "value": "Physicist"
              },
              {
                "key": "nationality",
                "value": "German"
              },
              {
                "key": "birth date",
                "value": "March 14, 1879"
              }
            ]
          },
          {
            "id": "Theory of Relativity",
            "type": "Theory",
            "properties": [
              {
                "key": "field of study",
                "value": "Physics"
              }
            ]
          }
        ],
        "rels": [
          {
            "source": {
              "id": "Albert Einstein",
              "type": "Physicist"
            },
            "target": {
              "id": "Theory of Relativity",
              "type": "Theory"
            },
            "type": "developed",
            "properties": [
                {
                  "key": "year",
                  "value": "1905"
                },
                {
                  "key": "best know for",
                  "value": "development"
                }
                {
                  "key": "published at",
                  "value": "On the Electrodynamics of Moving Bodies"
                }
              ]
          }
        ]
    }


4. revisit empty relationships and make sure they are not empty. there should be at least one relationship per verb.
5. ensure that no keys nor values are lists. Should that occur, create new objects with one each.
6. ensure that the output is a valid JSON object.
7. enrich the properties of both nodes and relationship so as to improve semantics, contextualization and accuracy; maintain the biomedical spe.

the input text is:

{{ unstructured_text }}

formatting instructions: {{ format_instructions }}

8. are you sure the result is correct? fix it
"""

extract_msg_1 = """
0. definitions
  - Property has only two attributes: key[string] and value[string].
  - Node has only three attributes: id[string], type[string] and properties[list[Property]].
  - Relationship has only four attributes: source[Node], target[Node], type[string] and properties[list[Property]].

  - treat copulas and verbs the same

for each full sentence, perform the step-by-step below:

1. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs.
2. group the adjectives and compound adjectives by their corresponding nouns and componound nouns.
3. classify the 2. adjectives and compound adjectives in the context of biomedical sciences
4. group the adverbs and compound adverbs by their corresponding verbs.
5. classify the 4. adverbs and compound adverbs in the context of biomedical sciences
6. group the verbs by the nouns they relate
7. create Node objects such that:
  - Node's ids are the nouns and compound nouns from step 2.
  - Node's properties are the corresponding nouns and compound nouns adjectives from step 2.
  - Node's  types are the enriched classes from step 3.
8. create Relationship objects such that:
  - Relationship's types are the leading verbs derived from step 4.
  - Relationship's properties are the adverbs and compound adverbs from step 5. that are related to the Relationship's type.
  - Relationship's sources and targets are the Nodes whose ids are the nouns and compound nound related by the Relationship's id verb.

9. example

  input:
  
      Albert Einstein is best known for developing the theory of relativity
  
  output:

    {
        "nodes": [
          {
            "id": "Albert Einstein",
            "type": "Person",
            "properties": [
              {
                "key": "role",
                "value": "Scientist"
              },
              {
                "key": "profession",
                "value": "Physicist"
              },
              {
                "key": "nationality",
                "value": "German"
              },
              {
                "key": "birth date",
                "value": "March 14, 1879"
              }
            ]
          },
          {
            "id": "Theory of Relativity",
            "type": "Theory",
            "properties": [
              {
                "key": "field of study",
                "value": "Physics"
              }
            ]
          }
        ],
        "rels": [
          {
            "source": {
              "id": "Albert Einstein",
              "type": "Physicist"
            },
            "target": {
              "id": "Theory of Relativity",
              "type": "Theory"
            },
            "type": "developed",
            "properties": [
                {
                  "key": "year",
                  "value": "1905"
                },
                {
                  "key": "best know for",
                  "value": "development"
                }
                {
                  "key": "published at",
                  "value": "On the Electrodynamics of Moving Bodies"
                }
              ]
          }
        ]
    }

10. return the result in a json object where the nodes and relationships are placed on a lists under `nodes` and `rels`, respectively; as in the step 9. example

the input text is:

{{ unstructured_text }}
{{ format_instructions }}

13. format the output as a valid JSON object.


"""

extract_msg_2 = """
0. definitions
  - Property has only two attributes: key[string] and value[string].
  - Node has only three attributes: id[string], type[string] and properties[list[Property]].
  - Relationship has only four attributes: source[Node], target[Node], type[string] and properties[list[Property]].

  - treat copulas and verbs the same

for each full sentence, perform the step-by-step below:

1. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs.
2. group the adjectives and compound adjectives by their corresponding nouns and compound nouns.
3. classify the 2. adjectives and compound adjectives in the context of biomedical sciences
4. group the adverbs and compound adverbs by their corresponding verbs.
5. classify the 4. adverbs and compound adverbs in the context of biomedical sciences
6. group the verbs by the nouns they relate
7. create Node objects such that:
  - Node's ids are the nouns and compound nouns from step 2.
  - Node's properties are the corresponding nouns and compound nouns adjectives from step 2.
  - Node's  types are the enriched classes from step 3.
8. create Relationship objects such that:
  - Relationship's types are the leading verbs derived from step 4.
  - Relationship's properties are the adverbs and compound adverbs from step 5. that are related to the Relationship's type.
  - Relationship's sources and targets are the Nodes whose ids are the nouns and compound nouns related by the Relationship's id verb.

9. format the output within a python dictionary such that:
  - Nodes are stored as a list in a key name `nodes`
  - Relationships are stored as a list in a key name `rels`

the input text is:

{{ unstructured_text }}
{{ format_instructions }}


"""

json_parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
prompt_semantic = PromptTemplate(
    template=extract_msg_1,
    input_variables=["unstructured_text"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
    format="json",
    template_format="jinja2",
    validate_template=True,
)


# punc_chain = {"unstructured_text": RunnablePassthrough()} | prompt_punctuation | llm | str_parser
# sent_chain = {"unstructured_text": RunnablePassthrough()} | prompt_sent | llm | lst_parser
sem_chain = {"unstructured_text": RunnablePassthrough()} | prompt_semantic | llm | json_parser

text = "Coronaviruses are enveloped, single stranded, positive sense RNA viruses."
print(text)
print()
# punc_data = punc_chain.invoke({"unstructured_text": text})
sem_data_1 = sem_chain.invoke({"unstructured_text": text})
# sent_data = sent_chain.invoke({"unstructured_text": punc_data})
print(sem_data_1)
# df = pd.read_csv("doc_clean_data_sentences_idx.csv")

# total = 0


# docs = df.groupby("descriptor")

# for doc in docs:
# while total < len(df):
#   print(total)
#   offset = 5
#   text = "\n".join(df.iloc[total:total + offset, 2].values)
#   sem_data = sem_chain.invoke({"unstructured_text": text})
#   # print(sem_data)
#   with open(f"data_gemini/{total}_{total+offset}.json", "w") as f:
#     f.write(sem_data)
#   total += 5

# sem_data = sem_chain.invoke({"unstructured_text": text})

# print(sem_data)
# "They are members of the Coronaviridae family."
# "Coronaviruses are subdivided into four genera: alpha coronaviruses, betacoronaviruses, gamma coronaviruses and delta coronaviruses (Biswas et al., 2020; Gorbalenya et al., 2020)."