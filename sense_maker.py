import re
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product

# from langchain.chat_models import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from typing import List, Optional

from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import (
    PydanticOutputParser,
    PandasDataFrameOutputParser,
    YamlOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.pydantic_v1 import Field, BaseModel, ConstrainedList


model = ChatOllama(model="mistral", temperature=0.0)


# parser = CommaSeparatedListOutputParser()\# prompt_message = """
# You will receive a tab separated value document that contains the extracted data from a PDF file's page.

# Task:
#     - Given a tabe separated document, group columns together that make more sense.
#     - the maximum number of groups is 2

# Format Instructions: {{ format_instructions }}
# Input: {{ unstructured_text }}
# """

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    # ResponseSchema(
    #     name="source",
    #     description="source used to answer the user's question, should be a website.",
    # ),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt_message = """
Rephrase and refine the given text to create a clear and coherent sentence.
Ensure that the key verbs and conjunctions from the original text are maintained.
Format Instructions: {{ format_instructions }}
Input: {{ unstructured_text }}

# Example Gibberish: "These results indicate —as expected-that
# measuring ionized calcium is useless for the early diagnosis of Vit-D
# insufficiency or deficiency (whilst PTH elevation seems to be an earlier

# 87.8

# 1.13-1.51 >1.51

# lonized calcium (mmol/L)

# Figure 9. Relationship between

# the PTH values and ionized calcium.

# V.-U. Hernando et al.

# predictor of such condition). "
# Example Cohesive Sentence: "The results indicate, as expected, that measuring ionized calcium is not effective for early diagnosis of vitamin D insufficiency or deficiency. However, elevated parathyroid hormone (PTH) levels emerge as a more predictive indicator for such conditions.
"""

prompt = PromptTemplate(
    template=prompt_message,
    input_variables=["unstructured_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    format="json",
    template_format="jinja2",
    validate_template=True,
)


chain = {"unstructured_text": RunnablePassthrough()} | prompt | model | parser

# input_folder = Path("groups")

# for i in input_folder.glob("*"):
#     with open(i, "r") as f:
#         data = f.read()
#     print(data)

#     rows = data.split("\n")[1:]

#     for row in rows:
#         print(row)
#         row_data = chain.invoke({"unstructured_text": row})
#         print(row_data)
#         print()

#     # f = Path(chunk.metadata.get("source")).stem
#     # df = parse(data)
#     # df.to_csv(graph_dir / f"{i}_{file_name}.csv", index=None)

text_1 = """These results indicate —as expected-that
measuring ionized calcium is useless for the early diagnosis of Vit-D
insufficiency or deficiency (whilst PTH elevation seems to be an earlier

87.8

1.13-1.51 >1.51

lonized calcium (mmol/L)

Figure 9. Relationship between

the PTH values and ionized calcium.

V.-U. Hernando et al.

predictor of such condition). 
"""

text_2 = """Socio-demographic and clinical baseline characteristics, sun exposure habits, and distribution of 25 OH Vit-D levels, ionized calcium, and PTH in the study

population.

Sex N (%)
Females 1148 (85.7%)
Males 191 (14.3%)
Median age (years) IQR

57 24

BMI (kg/m”) N (%)

Low weight 18 (1.34%)

Normal weight
Overweight
Obesity

364 (27.18%)
761 (56.83%)
196 (14.64%)

Sun exposure

N (%)

<30 min/week
>30 min/week

853 (63.7%)
486 (36.3%)

Use of sunscreen

N (%)

<3 times/week

>3 times/week

763 (57%)
576 (43%)

Distribution of 25 OH Vit-D levels

N (%)

Optimum 764 (57.1%)
Insufficient 386 (28.8%)
Deficient 189 (14.1%)
25 OH Vit-D Median (ng/mL) IQR

32.3 23.2

PTH Median (pg/mL) IQR

44.3 44.5

Ionized calcium Median (mmol/L) IQR

1.37 0.08
Socio-Economic Level (SEL) Classification and distribution N (%)

Low 112 (8.36%)
Middle 933 (69.68%)
High 294 (21.96%)

Abbreviations: SEL: Socio-economic level; BMI: body mass index; IQR: interquartile range; PTH: parathyroid hormone; Vit-D: vitamin D.

3.3. Distribution of the levels of 25 OH Vit-D, based on socio-demographic
characteristics, sun exposure habits, PTH and ionized calcium

The highest proportion of individuals with 25 OH Vit-D levels defined
as “deficient” (<20 ng/mL) or insufficient (20-29.9 ng/mL)” were within
the 51-70-year-old age range, belonged to the middle SEL, and were
overweight. >89% of the participants with deficiency or insufficiency

were exposed to sunlight <30 min per week, and >85% used some kind
of sunscreen."""
