from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import (
    PydanticOutputParser,
    PandasDataFrameOutputParser,
    YamlOutputParser,
    CommaSeparatedListOutputParser,
    # ListOutputParser,
)
import nltk
from langchain_community.chat_models import ChatOllama
from pathlib import Path
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
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import spacy


# Load the English language model
nlp = spacy.load("en_core_web_sm")

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


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

ollm = ChatOllama(model="mistral", temperature=0)

extract_msg = """
Given the input text below:

{{ unstructured_text }}

0. correct the punctuation.
1. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs:
2. organize 2.'s the schema below such that:
  - properties are objects with attributes key[string] and value[string]. neither key nor value can contain `,`
  - Property value is a compound noun, adjective or compound adjective and must not be inferred
  - Property keys is the inferred type of the property value in the biomedical context
  - nodes are objects with attributes id[string], type[string] and properties[list[Property]]
  - nodes are nouns whose properties are compounds nouns, adjectives and compound adjectives
  - Node id is noun or compound noun not a verb or compound adjective used as a noun
  - Node type is the inferred type of the property value in the biomedical context
  - Node properties is a list properties that are linked to the Node`s id
  - relationships are objects with attributes source[Node], target[Node], type[string] and properties[list[Property]]
  - Relationship type is the verb that describes the interaction between source and target in the correct direction
  - Relationship properties are the present adverbs and compound adjectives as well as the inferred meaningful properties
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


formatting instructions: {{ format_instructions }}

8. are you sure the result is correct? fix it
"""

extract_msg_1 = """

0. definitions
  - Property has only two attributes: key[string] and value[alpha numeric string], all in camel-case.
  - Node has only three attributes: id[string], type[string] and properties[list[Property]], all in camel-case.
  - Relationship has only four attributes: source[Node], target[Node], type[string] and properties[list[Property]], all in camel-case.
  - treat copulas and verbs the same
  - all values for all keys must be alpha numeric string
  - draw special attention to the verbs. each verb in the text must have a corresponding Relationship

given the input text below:

{{ unstructured_text }}


1. correct the punctuation
2. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs
3. group the adjectives and compound adjectives by their corresponding nouns and compounds nouns
4. classify step 3's adjectives and compound adjectives in the context of biomedical sciences
5. group the adverbs and compound adverbs by their corresponding verbs
6. classify step 5's adverbs and compound adverbs in the context of biomedical sciences
6. group the verbs by the nouns they relate

8. you will create Node objects:
  - Node's ids are the nouns and compound nouns identified at step 2
  - Node's properties are the corresponding nouns and compound nouns adjectives from step 3
  - Node's types are the enriched classes from step 5 
  - carefully enrich the properties with biomedical contextual cues for enhancing semantics

9. you will create Relationship objects:
  - Relationship's types are the leading verbs identified at step 2
  - Relationship's properties are the adverbs and compound adverbs from step 5 that are related to the Relationship's type
  - Relationship's sources and targets are the Nodes whose ids are the nouns and compound nouns related by the Relationship's id verb
  - Relationship's properties are different than those of its source and target.
  - carefully enrich the properties with biomedical contextual cues for enhancing semantics.

10. examples
- example a.

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
                }
              ]
            },
            {
              "id": "Theory of Relativity",
              "type": "Theory",
              "properties": [
                {
                  "key": "fieldOfStudy",
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
                    "key": "isBestKnownFor",
                    "value": "development"
                  }
                ]
            }
          ]
      }

  rationale:
    - the input contains two nouns; Albert Einstein and Theory of Relativity
    - the input contains two verbs; is and developed
    - the relationship type is developed, the source is Albert Einstein and the target is Theory of Relativity

- example b.

  input:
  
      Tomatoes are red, rounded, edible Fruits.
  
  output:

      {
          "nodes": [
            {
              "id": "Tomatoes",
              "type": "Fruit",
              "properties": [
                {
                  "key": "color",
                  "value": "red"
                },
                {
                  "key": "shape",
                  "value": "round"
                },
                {
                  "key": "function",
                  "value": "eat"
                },
                {
                  "key": "isMultipleOf",
                  "value": "tomato"
                }
              ]
            },
            {
              "id": "Fruits",
              "type": "Vegetable",
              "properties": [
                {
                  "key": "biologicalClassification",
                  "value": "fruit"
                },
                {
                  "key": "nutritionalClassification",
                  "value": "vegetable"
                },
                {
                  "key": "isMultipleOf",
                  "value": "fruit"
                }
              ]
            }
          ],
          "rels": [
            {
              "source": {
                "id": "Tomatoes",
                "type": "Fruit"
              },
              "target": {
                "id": "Fruits",
                "type": "biologicalClassification"
              },
              "type": "are",
              "properties": [
                  {
                    "key": "isTypeOf",
                    "value": "Fruit"
                  }
                ]
            }
          ]
      }

  rationale:
    - the input contains two nouns; Tomatoes and Fruits
    - the input contains one verb; is
    - the relationship type is is, the source is Tomatoes and the target is Fruits

11. the result is a JSON object, similar to step 10's a and b examples

12. double checking
  - if a relationship type is None, run steps 2 and 4 again
  - if a relationship source is None, run step 6 again
  - if a relationship target is None, run step 6 again
  - if a relationship's source or target is None, remove the relationship
  - if rels is an empty list and nodes a non empty list, run step 2 again
  - if there is not a relationship for every two nodes, run step 2 again
  - if the text contain two nouns or compound-nouns and one verb, the output must contain two nodes and one relationship
  - all values are alpha numeric strings

13. revisit empty relationships and make sure they are not empty. there should be at least one relationship per verb.
14. ensure that no keys nor values are lists. if that occurs, create new objects with one each.
15. ensure that the output is a valid JSON object.
16. enrich the properties of both nodes and relationship so as to improve semantics, contextualization and accuracy; maintain the biomedical domain
17. make the output JSON valid

the formatting instructions are: 

{{ format_instructions }}

18. are you sure the result is correct? fix it
"""


extract_msg_1 = """

0. definitions
  - Property has only two attributes: key[string] and value[alpha numeric string], all in camel-case.
  - Node has only three attributes: id[string], type[string] and properties[list[Property]], all in camel-case.
  - Relationship has only four attributes: source[Node], target[Node], type[string] and properties[list[Property]], all in camel-case.
  - Nodes represent entities
  - treat copulas and verbs the same
  - all values for all keys must be alpha numeric string
  - draw special attention to the verbs. each verb in the text must have a corresponding Relationship

given the input text below:

{{ unstructured_text }}


1. correct the punctuation
2. identify the entities and underlying relationships
 - entities are nouns, compound nouns that refer to, for example, Person, Substance, Dosage
 - entities properties are described by the corresponding adjectives and compound adjectives
 - relationships are verbs that relate a pair of entities
 - relationships properties are adverbs related to the corresponding verb

3. group the adjectives and compound adjectives by their corresponding nouns and compounds nouns
4. classify step 3's adjectives and compound adjectives in the context of biomedical sciences
5. group the adverbs and compound adverbs by their corresponding verbs
6. classify step 5's adverbs and compound adverbs in the context of biomedical sciences
6. group the verbs by the nouns they relate

8. you will create entity(Node) objects:
  - Node's ids are the nouns and compound nouns identified at step 2
  - Node's properties are the corresponding nouns and compound nouns adjectives from step 3
  - Node's types are the enriched classes from step 5 
  - carefully enrich the properties with biomedical contextual cues for enhancing semantics

9. you will create Relationship objects:
  - Relationship's types are the leading verbs identified at step 2
  - Relationship's properties are the adverbs and compound adverbs from step 5 that are related to the Relationship's type
  - Relationship's sources and targets are the Nodes whose ids are the nouns and compound nouns related by the Relationship's id verb
  - Relationship's properties are different than those of its source and target.
  - carefully enrich the properties with biomedical contextual cues for enhancing semantics.

10. examples
- example a.

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
                }
              ]
            },
            {
              "id": "Theory of Relativity",
              "type": "Theory",
              "properties": [
                {
                  "key": "fieldOfStudy",
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
                    "key": "isBestKnownFor",
                    "value": "development"
                  }
                ]
            }
          ]
      }

  rationale:
    - the input contains two nouns; Albert Einstein and Theory of Relativity
    - the input contains two verbs; is and developed
    - the relationship type is developed, the source is Albert Einstein and the target is Theory of Relativity

- example b.

  input:
  
      Tomatoes are red, rounded, edible Fruits.
  
  output:

      {
          "nodes": [
            {
              "id": "Tomatoes",
              "type": "Fruit",
              "properties": [
                {
                  "key": "color",
                  "value": "red"
                },
                {
                  "key": "shape",
                  "value": "round"
                },
                {
                  "key": "function",
                  "value": "eat"
                },
                {
                  "key": "isMultipleOf",
                  "value": "tomato"
                }
              ]
            },
            {
              "id": "Fruits",
              "type": "Vegetable",
              "properties": [
                {
                  "key": "biologicalClassification",
                  "value": "fruit"
                },
                {
                  "key": "nutritionalClassification",
                  "value": "vegetable"
                },
                {
                  "key": "isMultipleOf",
                  "value": "fruit"
                }
              ]
            }
          ],
          "rels": [
            {
              "source": {
                "id": "Tomatoes",
                "type": "Fruit"
              },
              "target": {
                "id": "Fruits",
                "type": "biologicalClassification"
              },
              "type": "are",
              "properties": [
                  {
                    "key": "isTypeOf",
                    "value": "Fruit"
                  }
                ]
            }
          ]
      }

  rationale:
    - the input contains two nouns; Tomatoes and Fruits
    - the input contains one verb; is
    - the relationship type is is, the source is Tomatoes and the target is Fruits


- example c.

  input:
  
      Coronaviruses are enveloped, single stranded, positive sense RNA viruses.

  
  output:

      {
          "nodes": [
            {
              "id": "Coronaviruses",
              "type": "Fruit",
              "properties": [
                {
                  "key": "structure",
                  "value": "enveloped"
                },
                {
                  "key": "morphology",
                  "value": "single stranded"
                },
                {
                  "key": "polarity",
                  "value": "positive sense"
                }
              ]
            },
            {
              "id": "viruses",
              "type": "RNA",
              "properties": [
                {
                  "key": "biologicalClassification",
                  "value": "fruit"
                },
                {
                  "key": "nutritionalClassification",
                  "value": "vegetable"
                },
                {
                  "key": "isMultipleOf",
                  "value": "fruit"
                }
              ]
            }
          ],
          "rels": [
            {
              "source": {
                "id": "Coronaviruses",
                "type": "Virus"
              },
              "target": {
                "id": "RNA Virus",
                "type": "Virus"
              },
              "type": "are",
              "properties": [
                  {
                    "key": "isTypeOf",
                    "value": "RNA virus"
                  }
                ]
            }
          ]
      }

  rationale:
    - the input contains one noun and one compound noun; Coronaviruses and RNA Viruses
    - the input contains one verb; are
    - the relationship type is `are`, the source is Coronaviruses and the target is RNA Viruses

- example d.
  input:

    authors declare no conﬂict of interest.
  
  output:
    
    {
        "nodes": [
            {
                "id": "Authors",
                "type": "Person",
                "properties": [
                    {
                        "key": "role",
                        "value": "Author"
                    }
                ]
            },
            {
                "id": "ConflictOfInterest",
                "type": "contract",
                "properties": [
                    {
                        "key": "activity",
                        "value": "Author"
                    }
                ]
            }
        ],
        "rels": [
            {
                "source": {
                    "id": "Authors",
                    "type": "Person"
                },
                "target": {
                    "id": "ConflictOfInterest",
                    "type": "contract"
                },
                "type": "declare",
                "properties": [
                    {
                        "key": "conflictOfInterest",
                        "value": "no"
                    }
                ]
            }
        ]
    }

- example e.

  input:

    John Doe: analyzed and interpreted the data; wrote the paper.
  
  output:
    
    {
        "nodes": [
            {
                "id": "John Doe",
                "type": "Person",
                "properties": [
                    {
                        "key": "role",
                        "value": "Author"
                    }
                ]
            },
            {
                "id": "Data",
                "type": "material",
                "properties": [
                    {
                        "key": "type",
                        "value": "digital"
                    }
                ]
            },
            {
                "id": "Paper",
                "type": "publication",
                "properties": [
                    {
                        "key": "publication",
                        "value": "Author"
                    }
                ]
            }
        ],
        "rels": [
            {
                "source": {
                    "id": "John Doe",
                    "type": "Person"
                },
                "target": {
                    "id": "Data",
                    "type": "material"
                },
                "type": "analyzed",
                "properties": [
                    {
                        "key": "verified",
                        "value": "data"
                    }
                ]
            },
            {
                "source": {
                    "id": "John Doe",
                    "type": "Person"
                },
                "target": {
                    "id": "Data",
                    "type": "material"
                },
                "type": "interpreted",
                "properties": [
                    {
                        "key": "madeSenseOf",
                        "value": "data"
                    }
                ]
            },
            {
                "source": {
                    "id": "John Doe",
                    "type": "Person"
                },
                "target": {
                    "id": "Paper",
                    "type": "publication"
                },
                "type": "wrote",
                "properties": [
                    {
                        "key": "authored",
                        "value": "paper"
                    }
                ]
            }
        ]
    }

11. the result is a JSON object, similar to step 10's a and b examples

12. double checking
  - if a relationship type is None, run steps 2 and 4 again
  - if a relationship source is None, run step 6 again
  - if a relationship target is None, run step 6 again
  - if a relationship's source or target is None, remove the relationship
  - if rels is an empty list and nodes a non empty list, run step 2 again
  - if there is not a relationship for every two nodes, run step 2 again
  - if the text contain two nouns or compound-nouns and one verb, the output must contain two nodes and one relationship
  - all values are alpha numeric strings

13. revisit empty relationships and make sure they are not empty. there should be at least one relationship per verb.
14. ensure that no keys nor values are lists. if that occurs, create new objects with one each.
15. ensure that the output is a valid JSON object.
16. enrich the properties of both nodes and relationship so as to improve semantics, contextualization and accuracy; maintain the biomedical domain
17. make the output JSON valid. if you cannot return a valid JSON object, return an empty KnowledgeGraph

the formatting instructions are: 

{{ format_instructions }}

18. are you sure the result is correct? fix it. if you cannot return a valid JSON object, return an empty KnowledgeGraph
19. if you cannot return a valid JSON object, return an empty KnowledgeGraph
20. if you encounter:
  1 -
    - error:
      rels -> 6 -> properties -> 0 -> value
        field required (type=value_error.missing)
    - solution:
      add a key name `value` and add value like `key` value

  2 -
    - error:
      nodes
        field required (type=value_error.missing)

    - solution:
      add an empty key named `nodes` with a empty list as value

  3 -
    - error:
      rels
        field required (type=value_error.missing)

    - solution:
      add an empty key named `rels` with a empty list as value

  4 -
    - error:
      rels -> integer -> target
        none is not an allowed value (type=type_error.none.not_allowed)

    - solution:
      create a target object with the properties' data

  5 -
    - error:
      rels -> integer -> type
        field required (type=value_error.missing)

    - solution:
      create a target object with the properties' data
          
  
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

camel_case = """
correctly the punctuation of the following input:
{{ format_instructions }}
- do no add any `\n` or enumerations to the output
- do not remove any words, symbols or notations from the input
{{ unstructured_text }}
"""


class Triple(BaseModel):
    """Generate a knowledge graph with entities and relationships."""

    sentence_type: str = Field(..., description="List of nodes in the knowledge graph")
    adverbs: List[str] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
    verbs: List[str] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
    subjects: List[str] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
    objects: List[str] = Field(
        ..., description="List of relationships in the knowledge graph"
    )
    modifiers: List[str] = Field(
        ..., description="List of relationships in the knowledge graph"
    )


camel_case = """
camel-case all nouns phrases, nouns, compound nouns, subjects and objects in the following text:
{{ format_instructions }}

examples:
  1.
    input: 
      I like Tomatoes
    ouput:
      I like Tomatoes
  2. 
    input:
      I like green Tomatoes
    output:
      I like greenTomatoes
  3.
    input:
      I like Brazilian Tomatoes
    output:
      I like brazilian Tomatoes
  4.
    input:
      I like POP Tomatoes
    output:
      I like POPTomatoes
  5.
    input:
      I like Tomatoe Juice
    output:
      I like TomatoJuice

{{ unstructured_text }}
"""

camel_case_g = """
given the following context:

{{ unstructured_text }}

perform the procedure below

1. correct the punctuation, spelling and typos.
2. camel-case exclusively all entities, e.g., noun-phrases, nouns, concepts, percentages, p-values, ranges, dosages, concentrations, techniques, methodologies, time periods, time ranges, age ranges, authors, scientific references and other clinical trial related parameters; consider the biomedical context, following the guidelines:
    a- an entity must represent a specific semantic meaning
    b- an entity must not start with a preposition
    c- an entity should not be too long
    d- a camel-cased entity must contain at most 3 nouns
    e- a camel-cased entity must not contain a adjectives
    f- all adjectives must be present in the output

3. snake-case exclusively all phrasal verbs, idiomatic verbs, idiomatic expressions, two consecutive verbs and modal verbs, following the guidelines:
    a- verbs must only be concatenated with other verbs, copulas and prepositions
    b- adverbs must not be concatenated with a noun
    c- a snake-cased verbs must contain no nouns nor articles

take all camel-cased words and pass it through the guidelines 2.a, 2.b, 2.c, 2.d, 2.e or 2.f, make all necessary corrections.
take all snake-cased words and pass it through the guidelines3.a, 3.b or 3.c, make all necessary corrections.

no words can be missing from the corrected context

give it a finish and output the transformed context
"""

camel_case_g = """
given the following context:

{{ unstructured_text }}

perform the procedure below:

1. correct the punctuation, spelling and typos.
2. camel-case exclusively all entities, e.g., noun-phrases, nouns, concepts, percentages, p-values, ranges, dosages, concentrations, techniques, latin terms and phrases, methodologies, time periods, time ranges, age ranges, authors, scientific references and other clinical trial related parameters; consider the biomedical context, following the guidelines:
    a- represent a semantic meaning
    b- not start with a preposition
    c- not be overly long (contain at most 3 nouns)
    e- all adjectives must be present in the output
    f- acronyms and initialisms related to diseases, viruses, and other scientific terms should be treated as single entities without separators (e.g., "COVID-19" becomes "Covid19", "SARS-CoV-2" becomes "SarsCov2",  "25(OH'D or 25(OH)D or 25 (OH' D" become "25OHD", )
    g- numerical parts of entities should be directly attached to the preceding text without spaces or hyphens
    h- apply camel-casing to compound nouns and adjectives to form a single entity without spaces, ensuring the first letter of each subsequent word is capitalized (e.g., "Vitamin D" becomes "VitaminD" and "fat soluble" becomes "fatSoluble").

3. snake-case exclusively all phrasal verbs, prepositional verbs, verb-particle constructions, inseparable phrasal verbs, idiomatic verbs, idiomatic expressions, two consecutive verbs and modal verbs, following the guidelines:
    a- verbs should only be concatenated with other surrounding verbs, copulas and prepositions
    b- adverbs must not be concatenated with a noun
    c- ensure that snake-cased verbs contain no nouns nor articles
    d- verb phrases should maintain readability and natural language flow
    e- avoid concatenating verbs in a manner that creates unwieldy or unnatural phrases. 
    f- or complex verb phrases where snake-casing would impair readability, maintain standard spelling and spacing.
    g- adverbs must not be concatenated with a noun
    h- use snake-casing for phrasal verbs by connecting all parts of the verb (including prepositions and adverbs) with underscores (e.g., "is thought of" becomes "is_thought_of")

have you overlooked any of the 2.a - 2.h guidelines? make all necessary corrections.
have you overlooked any of the 3.a - 3.h guidelines? make all necessary corrections.

no words can be removed from the corrected context
no words can be added to the corrected context

ensure all phrasal verbs are snake-cased
ensure all prepositional verbs are snake-cased
ensure all verb-particle constructions are snake-cased
ensure all inseparable phrasal verbs are snake-cased
ensure all idiomatic verbs are snake-cased
ensure all idiomatic expressions are snake-cased
ensure all consecutive verbs are snake-cased
ensure all modal verbs are snake-cased

ensure all noun-phrases are camel-cased
ensure all nouns are camel-cased
ensure all concepts are camel-cased
ensure all percentages are camel-cased
ensure all p-values are camel-cased
ensure all ranges are camel-cased
ensure all dosages are camel-cased
ensure all concentrations are camel-cased
ensure all techniques are camel-cased
ensure all methodologies are camel-cased
ensure all time periods are camel-cased
ensure all time ranges are camel-cased
ensure all latin terms and phrases are camel-cased
ensure all age ranges are camel-cased
ensure all authors are camel-cased
ensure all scientific references are camel-cased

{{ format_instructions }}
"""

camel_case_f = """
given the following context:

{{ unstructured_text }}

correct the punctuation, spelling and typos.
ensure that no word is removed from the input text.
ensure that no word is added to the input text

is the output correct?
{{ format_instructions }}
"""


def prompt_it(data):
  data = f"""given the text below:

{data}

correct the punctuation, spelling, orthography and typos.
then, extract the meaningful and complete sentences in the same order they appear on the text.
place the spurious text on another section called `trash:`.
the trash section should succeed a line-break from the good text.
extract the text that prevents meaningful, human readable sentences to form. 
the corrected text should contain no line breaks, no symbols that are not present in the original text.
the corrected text is a standard text file, no markdown language should be present.

the trash section should contain all:
- text that do not contribute to the formation of semantic, complete and meaningful sentences.
- table or tabular data, e.g.: 
- text that is detrimental for the contextual understanding of whole cohesive, meaningful sentences.

some samples include:

1-
input text:
  
Vitamin D has various effects during pregnancy such as placental implantation, angiogenesis, immune function, oxidative stress, endothelial function, inflammatory response, and glucose homeostasis * Corresponding author at: 1604th Street, No. 9, Cankaya, Ankara 06800, Turkey. E-mail address: selcansinaci@gmail.com (S. Sinaci). https://doi.org/10.1016/j.jsbmb.2021.105964 Received 11 May 2021; Received in revised form 4 August 2021; Accepted 9 August 2021 Availableonline11August2021 0960-0760/©2021ElsevierLtd.Allrightsreserved. S. Sinaci et al. [12,13].

output text:

Vitamin D has various effects during pregnancy such as placental implantation, angiogenesis, immune function, oxidative stress, endothelial function, inflammatory response, and glucose homeostasis 

trash:
* Corresponding author at: 1604th Street, No. 9, Cankaya, Ankara 06800, Turkey. E-mail address: selcansinaci@gmail.com (S. Sinaci). https://doi.org/10.1016/j.jsbmb.2021.105964 Received 11 May 2021; Received in revised form 4 August 2021; Accepted 9 August 2021 Availableonline11August2021 0960-0760/©2021ElsevierLtd.Allrightsreserved. S. Sinaci et al. [12,13].

2-
input text:

However, large dose of zinc can cause the suppression of immune system and interfere with absorption of other minerals. The tolerable upper intake level established by the Food and Nutrition Board of the US institute of Medicine is 40 mg a day and tolerable upper intake level from dietary reference intakes (DRIs) for Koreans is 35 mg/day. https://e-nrp.org	https://doi.org/10.4162/nrp.2021.15.S1.S1	S10 COVID-19 and immunomodulatory nutrients IRON Effects of iron on immune function and viral infection Iron is an essential trace element that functions in diverse metabolic processes including oxygen transport, electron transport chain, and DNA synthesis.

output text:

However, large dose of zinc can cause the suppression of immune system and interfere with absorption of other minerals. The tolerable upper intake level established by the Food and Nutrition Board of the US institute of Medicine is 40 mg a day and tolerable upper intake level from dietary reference intakes (DRIs) for Koreans is 35 mg/day. COVID-19 and immunomodulatory nutrients IRON Effects of iron on immune function and viral infection Iron is an essential trace element that functions in diverse metabolic processes including oxygen transport, electron transport chain, and DNA synthesis.

trash:
https://e-nrp.org	https://doi.org/10.4162/nrp.2021.15.S1.S1	S10 

3-
input text:

Positive correlations were observed between vitamin D levels and the lymphocyte and leukocyte counts (r = 0.294, p=0.001; r = 0.143, Tab. 3. The relationship between vitamin D status and other parameters. Age (year) Gender; n (%) Ferritin (μgr/L) CRP (mg/L) D-dimer (ug/L) ALT (U/L) AST (U/L) Leukocyte (×103/mm3) Lymphocyte (×103/mm3) Platelet (×103/mm3) Min–Max (Median) Mean±Sd Female Male Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Deﬁ cient(n=85) 24–88 (61) 59.18±16.32  8.4–4721 (174.9) 356.60±605.46 0.4–372.2 (72.5) 91.79±86.33 111–8720 (745) 1375.94±1668.71 4–246 (19) 27.53±30.67 7–157 (22) 30.05±21.43 2.4–22.7 (6.8) 7.08±3.02 0.3–3.3 (1.3) 1.42±0.64 35–577 (223) 235.71±99.34 Vitamin D status Insufﬁcient (n=94) 23–94 (57) 57.37±19.46  2.2–4260 (109.7) 256.35±507.75 0.4–355.7 (26.4) 60.27±71.75 112–8840 (742) 1061.88±1258.33 5–184 (25.5) 35.78±33.73 9–112 (24) 30.77±19.95 2.6–76.5 (7.6) 8.79±7.73 0.2–7.3 (1.5) 1.70±1.00 4.7–925 (242.5) 263.34±115.62 Normal (n=25) 28–82 (50) 52.72±16.51  9.1–811.7 (81.2) 153.54±186.66 0.8–193.5 (4.8) 35.01±59.83 80–2760 (630) 660.88±582.59 10–105 (20) 28.52±25.17 12–47 (20) 23.92±8.43 3.7–16 (7.5) 7.93±2.67 0.7–3.5 (2) 2.01±0.72 134–421 (261) 262.88±78.02 p d0.283 b0.601 e0.034* e0.001** e0.054 e0.099 e0.593 e0.043* e0.002** e0.158 dOneway ANOVATest, eKruskal–Wallis Test, **p<0.01, *p<0.05 Basaran N et al. The relationship between vitamin D and the severity of COVID-19 Tab. 4. Correlation analysis (r) between vitamin D and other labora-tory parameters. xx p=0.042, respectively) (Fig. 2 D).

output text:

Positive correlations were observed between vitamin D levels and the lymphocyte and leukocyte counts (r = 0.294, p=0.001; r = 0.143, Tab. 3. The relationship between vitamin D status and other parameters. Correlation analysis (r) between vitamin D and other labora-tory parameters. xx p=0.042, respectively) (Fig. 2 D).

trash: 
Age (year) Gender; n (%) Ferritin (μgr/L) CRP (mg/L) D-dimer (ug/L) ALT (U/L) AST (U/L) Leukocyte (×103/mm3) Lymphocyte (×103/mm3) Platelet (×103/mm3) Min–Max (Median) Mean±Sd Female Male Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Deﬁ cient(n=85) 24–88 (61) 59.18±16.32  8.4–4721 (174.9) 356.60±605.46 0.4–372.2 (72.5) 91.79±86.33 111–8720 (745) 1375.94±1668.71 4–246 (19) 27.53±30.67 7–157 (22) 30.05±21.43 2.4–22.7 (6.8) 7.08±3.02 0.3–3.3 (1.3) 1.42±0.64 35–577 (223) 235.71±99.34 Vitamin D status Insufﬁcient (n=94) 23–94 (57) 57.37±19.46  2.2–4260 (109.7) 256.35±507.75 0.4–355.7 (26.4) 60.27±71.75 112–8840 (742) 1061.88±1258.33 5–184 (25.5) 35.78±33.73 9–112 (24) 30.77±19.95 2.6–76.5 (7.6) 8.79±7.73 0.2–7.3 (1.5) 1.70±1.00 4.7–925 (242.5) 263.34±115.62 Normal (n=25) 28–82 (50) 52.72±16.51  9.1–811.7 (81.2) 153.54±186.66 0.8–193.5 (4.8) 35.01±59.83 80–2760 (630) 660.88±582.59 10–105 (20) 28.52±25.17 12–47 (20) 23.92±8.43 3.7–16 (7.5) 7.93±2.67 0.7–3.5 (2) 2.01±0.72 134–421 (261) 262.88±78.02 p d0.283 b0.601 e0.034* e0.001** e0.054 e0.099 e0.593 e0.043* e0.002** e0.158 dOneway ANOVATest, eKruskal–Wallis Test, **p<0.01, *p<0.05 Basaran N et al. The relationship between vitamin D and the severity of COVID-19 Tab. 4. 

4- 
input text:

Among different age groups, the prevalence of Vitamin D deficiency was high in 20–29 (68.08%) years with a mean level of 14.60 ng/mL ± 3.84 and 30–39 (60.97%) years with a mean level of 14.07 ng/mL ± 4.16. Vitamin D insufficiency was also reported high among these age groups, 95.74%, and 92.68%, respec-tively. The study reported 65.52% deficiency and 93.10% insufficiency of Vitamin D among the overweight group cases (Table 3).

output text:

Among different age groups, the prevalence of Vitamin D deficiency was high in 20–29 (68.08%) years with a mean level of 14.60 ng/mL ± 3.84 and 30–39 (60.97%) years with a mean level of 14.07 ng/mL ± 4.16. Vitamin D insufficiency was also reported high among these age groups, 95.74%, and 92.68%, respec-tively. The study reported 65.52% deficiency and 93.10% insufficiency of Vitamin D among the overweight group cases (Table 3).

trash:

5- 

input text:

Interestingly, patients suffering from hypertension were more prone to      Vitamin D treatment has been linked to decreased Table 5 Association of Various Risk Factors for Mortality Among Cases Risk Factor	Death (%) N=18	No Death (%) N=138	P value	Chi Square	OR Sex Age BMI Alcohol Smoking DM HTN Vit D level Male Female <50 ≥50 Normal (<23 kg/m2) Overweight (23–24.9 kg/m2) Obese (≥ 25 kg/m2) Yes No Yes No Yes No Yes No Deficient (<20 ng/mL) Insufficient (20–30 ng/mL) Normal (>30 ng/mL)  0.1638 (>0.05)	  (<0.00001)	  (>0.05)	2.98 0.1812 (>0.05)	  (>0.05)	  (<0.05)	  (<0.0001)	  (>0.05)	1.06 2528	https://doi.org/10.2147/IJGM.S309003	International Journal of General Medicine 2021:14 DovePress Dovepress angiotensin II and renin levels in multiple studies.40–42 Our Singh et al have an increased expression of ACE2; therefore, ACE data show a significant role of hypertension among	inhibitors are prescribed to regulate their role in disease COVID-19 patients with increased mortality, which can be reduced with Vitamin D intake.

output text:

Interestingly, patients suffering from hypertension were more prone to have an increased expression of ACE2; therefore, ACE data show a significant role of hypertension among	inhibitors are prescribed to regulate their role in disease COVID-19 patients with increased mortality, which can be reduced with Vitamin D intake.

trash:
      Vitamin D treatment has been linked to decreased Table 5 Association of Various Risk Factors for Mortality Among Cases Risk Factor	Death (%) N=18	No Death (%) N=138	P value	Chi Square	OR Sex Age BMI Alcohol Smoking DM HTN Vit D level Male Female <50 ≥50 Normal (<23 kg/m2) Overweight (23–24.9 kg/m2) Obese (≥ 25 kg/m2) Yes No Yes No Yes No Yes No Deficient (<20 ng/mL) Insufficient (20–30 ng/mL) Normal (>30 ng/mL)  0.1638 (>0.05)	  (<0.00001)	  (>0.05)	2.98 0.1812 (>0.05)	  (>0.05)	  (<0.05)	  (<0.0001)	  (>0.05)	1.06 2528	https://doi.org/10.2147/IJGM.S309003	International Journal of General Medicine 2021:14 DovePress Dovepress angiotensin II and renin levels in multiple studies.40–42 Our Singh et al 

ensure that no information removed nor added, but re-arranged.
ensure that the corrected text contains no typos, no spelling errors and no orthographic errors
ensure that the corrected text contains only complete, cohesive and correct sentences.
ensure that the corrected text contains no line breaks
"""
  return data

# You are an assistant that will help parsing a messing text into well formed, correct sentences from the input text
fix_text_prompt = """given the text below:

{{ unstructured_text }}

correct the punctuation, spelling, orthography and typos.
then, extract the meaningful and complete sentences in the same order they appear on the text.
place the spurious text on another section called `trash:`.
the trash section should succeed a line-break from the good text.
extract the text that prevents meaningful, human readable sentences to form. 
the corrected text should contain no line breaks, no symbols that are not present in the original text.
the corrected text is a standard text file, no markdown language should be present.

the trash section should contain all:
- text that do not contribute to the formation of semantic, complete and meaningful sentences.
- table or tabular data, e.g.: 
- text that is detrimental for the contextual understanding of whole cohesive, meaningful sentences.

some samples include:

1-
input text:
  
Vitamin D has various effects during pregnancy such as placental implantation, angiogenesis, immune function, oxidative stress, endothelial function, inflammatory response, and glucose homeostasis * Corresponding author at: 1604th Street, No. 9, Cankaya, Ankara 06800, Turkey. E-mail address: selcansinaci@gmail.com (S. Sinaci). https://doi.org/10.1016/j.jsbmb.2021.105964 Received 11 May 2021; Received in revised form 4 August 2021; Accepted 9 August 2021 Availableonline11August2021 0960-0760/©2021ElsevierLtd.Allrightsreserved. S. Sinaci et al. [12,13].

output text:

Vitamin D has various effects during pregnancy such as placental implantation, angiogenesis, immune function, oxidative stress, endothelial function, inflammatory response, and glucose homeostasis 

trash:
* Corresponding author at: 1604th Street, No. 9, Cankaya, Ankara 06800, Turkey. E-mail address: selcansinaci@gmail.com (S. Sinaci). https://doi.org/10.1016/j.jsbmb.2021.105964 Received 11 May 2021; Received in revised form 4 August 2021; Accepted 9 August 2021 Availableonline11August2021 0960-0760/©2021ElsevierLtd.Allrightsreserved. S. Sinaci et al. [12,13].

2-
input text:

However, large dose of zinc can cause the suppression of immune system and interfere with absorption of other minerals. The tolerable upper intake level established by the Food and Nutrition Board of the US institute of Medicine is 40 mg a day and tolerable upper intake level from dietary reference intakes (DRIs) for Koreans is 35 mg/day. https://e-nrp.org	https://doi.org/10.4162/nrp.2021.15.S1.S1	S10 COVID-19 and immunomodulatory nutrients IRON Effects of iron on immune function and viral infection Iron is an essential trace element that functions in diverse metabolic processes including oxygen transport, electron transport chain, and DNA synthesis.

output text:

However, large dose of zinc can cause the suppression of immune system and interfere with absorption of other minerals. The tolerable upper intake level established by the Food and Nutrition Board of the US institute of Medicine is 40 mg a day and tolerable upper intake level from dietary reference intakes (DRIs) for Koreans is 35 mg/day. COVID-19 and immunomodulatory nutrients IRON Effects of iron on immune function and viral infection Iron is an essential trace element that functions in diverse metabolic processes including oxygen transport, electron transport chain, and DNA synthesis.

trash:
https://e-nrp.org	https://doi.org/10.4162/nrp.2021.15.S1.S1	S10 

3-
input text:

Positive correlations were observed between vitamin D levels and the lymphocyte and leukocyte counts (r = 0.294, p=0.001; r = 0.143, Tab. 3. The relationship between vitamin D status and other parameters. Age (year) Gender; n (%) Ferritin (μgr/L) CRP (mg/L) D-dimer (ug/L) ALT (U/L) AST (U/L) Leukocyte (×103/mm3) Lymphocyte (×103/mm3) Platelet (×103/mm3) Min–Max (Median) Mean±Sd Female Male Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Deﬁ cient(n=85) 24–88 (61) 59.18±16.32  8.4–4721 (174.9) 356.60±605.46 0.4–372.2 (72.5) 91.79±86.33 111–8720 (745) 1375.94±1668.71 4–246 (19) 27.53±30.67 7–157 (22) 30.05±21.43 2.4–22.7 (6.8) 7.08±3.02 0.3–3.3 (1.3) 1.42±0.64 35–577 (223) 235.71±99.34 Vitamin D status Insufﬁcient (n=94) 23–94 (57) 57.37±19.46  2.2–4260 (109.7) 256.35±507.75 0.4–355.7 (26.4) 60.27±71.75 112–8840 (742) 1061.88±1258.33 5–184 (25.5) 35.78±33.73 9–112 (24) 30.77±19.95 2.6–76.5 (7.6) 8.79±7.73 0.2–7.3 (1.5) 1.70±1.00 4.7–925 (242.5) 263.34±115.62 Normal (n=25) 28–82 (50) 52.72±16.51  9.1–811.7 (81.2) 153.54±186.66 0.8–193.5 (4.8) 35.01±59.83 80–2760 (630) 660.88±582.59 10–105 (20) 28.52±25.17 12–47 (20) 23.92±8.43 3.7–16 (7.5) 7.93±2.67 0.7–3.5 (2) 2.01±0.72 134–421 (261) 262.88±78.02 p d0.283 b0.601 e0.034* e0.001** e0.054 e0.099 e0.593 e0.043* e0.002** e0.158 dOneway ANOVATest, eKruskal–Wallis Test, **p<0.01, *p<0.05 Basaran N et al. The relationship between vitamin D and the severity of COVID-19 Tab. 4. Correlation analysis (r) between vitamin D and other labora-tory parameters. xx p=0.042, respectively) (Fig. 2 D).

output text:

Positive correlations were observed between vitamin D levels and the lymphocyte and leukocyte counts (r = 0.294, p=0.001; r = 0.143, Tab. 3. The relationship between vitamin D status and other parameters. Correlation analysis (r) between vitamin D and other labora-tory parameters. xx p=0.042, respectively) (Fig. 2 D).

trash: 
Age (year) Gender; n (%) Ferritin (μgr/L) CRP (mg/L) D-dimer (ug/L) ALT (U/L) AST (U/L) Leukocyte (×103/mm3) Lymphocyte (×103/mm3) Platelet (×103/mm3) Min–Max (Median) Mean±Sd Female Male Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Min–Max (Median) Mean±Sd Deﬁ cient(n=85) 24–88 (61) 59.18±16.32  8.4–4721 (174.9) 356.60±605.46 0.4–372.2 (72.5) 91.79±86.33 111–8720 (745) 1375.94±1668.71 4–246 (19) 27.53±30.67 7–157 (22) 30.05±21.43 2.4–22.7 (6.8) 7.08±3.02 0.3–3.3 (1.3) 1.42±0.64 35–577 (223) 235.71±99.34 Vitamin D status Insufﬁcient (n=94) 23–94 (57) 57.37±19.46  2.2–4260 (109.7) 256.35±507.75 0.4–355.7 (26.4) 60.27±71.75 112–8840 (742) 1061.88±1258.33 5–184 (25.5) 35.78±33.73 9–112 (24) 30.77±19.95 2.6–76.5 (7.6) 8.79±7.73 0.2–7.3 (1.5) 1.70±1.00 4.7–925 (242.5) 263.34±115.62 Normal (n=25) 28–82 (50) 52.72±16.51  9.1–811.7 (81.2) 153.54±186.66 0.8–193.5 (4.8) 35.01±59.83 80–2760 (630) 660.88±582.59 10–105 (20) 28.52±25.17 12–47 (20) 23.92±8.43 3.7–16 (7.5) 7.93±2.67 0.7–3.5 (2) 2.01±0.72 134–421 (261) 262.88±78.02 p d0.283 b0.601 e0.034* e0.001** e0.054 e0.099 e0.593 e0.043* e0.002** e0.158 dOneway ANOVATest, eKruskal–Wallis Test, **p<0.01, *p<0.05 Basaran N et al. The relationship between vitamin D and the severity of COVID-19 Tab. 4. 

4- 
input text:

Among different age groups, the prevalence of Vitamin D deficiency was high in 20–29 (68.08%) years with a mean level of 14.60 ng/mL ± 3.84 and 30–39 (60.97%) years with a mean level of 14.07 ng/mL ± 4.16. Vitamin D insufficiency was also reported high among these age groups, 95.74%, and 92.68%, respec-tively. The study reported 65.52% deficiency and 93.10% insufficiency of Vitamin D among the overweight group cases (Table 3).

output text:

Among different age groups, the prevalence of Vitamin D deficiency was high in 20–29 (68.08%) years with a mean level of 14.60 ng/mL ± 3.84 and 30–39 (60.97%) years with a mean level of 14.07 ng/mL ± 4.16. Vitamin D insufficiency was also reported high among these age groups, 95.74%, and 92.68%, respec-tively. The study reported 65.52% deficiency and 93.10% insufficiency of Vitamin D among the overweight group cases (Table 3).

trash:

5- 

input text:

Interestingly, patients suffering from hypertension were more prone to      Vitamin D treatment has been linked to decreased Table 5 Association of Various Risk Factors for Mortality Among Cases Risk Factor	Death (%) N=18	No Death (%) N=138	P value	Chi Square	OR Sex Age BMI Alcohol Smoking DM HTN Vit D level Male Female <50 ≥50 Normal (<23 kg/m2) Overweight (23–24.9 kg/m2) Obese (≥ 25 kg/m2) Yes No Yes No Yes No Yes No Deficient (<20 ng/mL) Insufficient (20–30 ng/mL) Normal (>30 ng/mL)  0.1638 (>0.05)	  (<0.00001)	  (>0.05)	2.98 0.1812 (>0.05)	  (>0.05)	  (<0.05)	  (<0.0001)	  (>0.05)	1.06 2528	https://doi.org/10.2147/IJGM.S309003	International Journal of General Medicine 2021:14 DovePress Dovepress angiotensin II and renin levels in multiple studies.40–42 Our Singh et al have an increased expression of ACE2; therefore, ACE data show a significant role of hypertension among	inhibitors are prescribed to regulate their role in disease COVID-19 patients with increased mortality, which can be reduced with Vitamin D intake.

output text:

Interestingly, patients suffering from hypertension were more prone to have an increased expression of ACE2; therefore, ACE data show a significant role of hypertension among	inhibitors are prescribed to regulate their role in disease COVID-19 patients with increased mortality, which can be reduced with Vitamin D intake.

trash:
      Vitamin D treatment has been linked to decreased Table 5 Association of Various Risk Factors for Mortality Among Cases Risk Factor	Death (%) N=18	No Death (%) N=138	P value	Chi Square	OR Sex Age BMI Alcohol Smoking DM HTN Vit D level Male Female <50 ≥50 Normal (<23 kg/m2) Overweight (23–24.9 kg/m2) Obese (≥ 25 kg/m2) Yes No Yes No Yes No Yes No Deficient (<20 ng/mL) Insufficient (20–30 ng/mL) Normal (>30 ng/mL)  0.1638 (>0.05)	  (<0.00001)	  (>0.05)	2.98 0.1812 (>0.05)	  (>0.05)	  (<0.05)	  (<0.0001)	  (>0.05)	1.06 2528	https://doi.org/10.2147/IJGM.S309003	International Journal of General Medicine 2021:14 DovePress Dovepress angiotensin II and renin levels in multiple studies.40–42 Our Singh et al 

ensure that no information removed nor added, but re-arranged.
ensure that the corrected text contains no typos, no spelling errors and no orthographic errors
ensure that the corrected text contains only complete, cohesive and correct sentences.
ensure that the corrected text contains no line breaks

{{ format_instructions }}
"""

def text_to_sentences(text):
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

response_schemas = [
    ResponseSchema(name="corrected_text", description="the correctly punctuated text.")
]
# json_parser = PydanticOutputParser(pydantic_object=Triple)

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
camel_case_prompt = PromptTemplate(
    template=fix_text_prompt,
    input_variables=["unstructured_text"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    format="json",
    template_format="jinja2",
    validate_template=True,
)
# punc_chain = {"unstructured_text": RunnablePassthrough()} | prompt_punctuation | llm | str_parser
# sent_chain = {"unstructured_text": RunnablePassthrough()} | prompt_sent | llm | lst_parser
# sem_chain = {"unstructured_text": RunnablePassthrough()} | prompt_semantic | llm | json_parser
# sem_chain = {"unstructured_text": RunnablePassthrough()} | prompt_splt | llm | list_parser

sem_chain = (
    {"unstructured_text": RunnablePassthrough()}
    | camel_case_prompt
    | llm
    | output_parser
)

# for text in texts:
#     print(text)
#     # sentences = nltk.sent_tokenize(text)
#     # print()
#     sem_data_1 = sem_chain.invoke({"unstructured_text": text})
#     print(sem_data_1)
#     # for sentence in sentences:
#     #   try:
#     #     print("S: ", sentence)
#     #   except Exception as e:
#     #     print(e, " on ", sentence)
#     print()
#     print()
# # df = pd.read_csv("doc_clean_data_sentences_idx.csv")


# input_files = Path("docx_2_text").glob("*.txt")

# for i_f in input_files:
#     # print(doc)
#     name = i_f.stem
#     # data = " ".join(doc["text"].values)
#     with open(i_f, "r") as f:
#         data = f.read()

#     sem_data = sem_chain.invoke({"unstructured_text": data})
#     print(sem_data)

#     with open(f"corrected_punctuation/{i_f.stem}.txt", "w") as f:
#         f.write(sem_data["corrected_text"])

#     print()
#     print()

# input_files = [Path("docx_2_text_1/0ddcfd1cb232030765e364e5064ddcb9.txt")]
input_folder = Path("docx_2_text_1")

input_files = input_folder.glob("*.txt")

chunk_size = 5000

for f in input_files:
    with open(f, "r") as fd:
        data = fd.read()


    # with open(f"prompt_gemini_full/{f.stem}.txt", "w") as fd:
    #     fd.write(prompt)


        data_sents = text_to_sentences(data)
        chunk = ""
        chunks = []
        s_sum = 0
        c_sum = 0
        for i, s in enumerate(data_sents):
            s_sum += len(s)
            if len(s) + len(chunk) <= chunk_size:
                chunk += " " + s
            else:
                # print(i, len(chunk))
                c_sum += len(chunk)
                chunks.append(chunk)
                chunk = s
        chunks.append(chunk)

        for block_id, c in enumerate(chunks):
          # try:
          #   sem_data = sem_chain.invoke({"unstructured_text": data})
          # except Exception as e:
          #   print(e, f)
          #   continue
          # output = sem_data["corrected_text"]
          data = prompt_it(c)
          with open(f"chunked_prompted_{chunk_size}/{f.stem}_{block_id}.txt", "w") as fd:
            fd.write(data)
                        
# chunked_folder = Path(f"chunked_{chunk_size}")
# chunked_files = chunked_folder.glob("*.txt")

# for f in chunked_files:

    # if Path(f"gemini_{chunk_size}/{f.stem}.txt").exists():
    #   continue

    # with open(f, "r") as fd:
    #   text = fd.read()
    # try:
    #   sem_data = sem_chain.invoke({"unstructured_text": text})
    # except Exception as e:
    #   print(e, f)
    #   continue
    # # print()
    # output = sem_data["corrected_text"]
    # with open(f"gemini_{chunk_size}/{f.stem}.txt", "w") as fd:
    #   text = fd.write(output)
    # if Path(f"gemini/{f.stem}.txt").exists():
    #   continue

    # # with open(f, "r") as fd:
    # # #   text = fd.read()
    # try:
    #   sem_data = sem_chain.invoke({"unstructured_text": data})
    # except Exception as e:
    #   print(e, f)
    #   continue
    # # print()
    # output = sem_data["corrected_text"]
    # # with open(f"gemini/{f.stem}.txt", "w") as fd:
    # #   text = fd.write(output)