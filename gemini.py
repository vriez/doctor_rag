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


extract_msg = """
take the input text and follow the procedure step by step bellow:
0. correct the punctuation, syntax and morphology of the input text.
1. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs:
2. organize 2.'s the schema below such that:
  - properties are objects with attributes key[string] and value[string]. neither key nor value can contain `,`
  - Property value is a compound noun, adjective or compound adjective and must not be infered
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
  - Property has only two attributes: key[string] and value[alpha numeric string], all in camel-case.
  - Node has only three attributes: id[alpha numeric string], type[string] and properties[list[Property]], all in camel-case.
  - Relationship has only four attributes: source[Node], target[Node], type[string] and properties[list[Property]], all in camel-case.
  - treat copulas and verbs the same
  - all values for all keys must be alpha numeric string
  - draw special attention to the verbs. each verb in the text must have a corresponding Relationship

1. identify the nouns, compound nouns, adjective and compound adjectives, adverbs and verbs.
2. group the adjectives and compound adjectives by their corresponding nouns and compounds nouns.
3. classify the 2. adjectives and compound adjectives in the context of biomedical sciences
4. group the adverbs and compound adverbs by their corresponding verbs.
5. classify the 4. adverbs and compound adverbs in the context of biomedical sciences
6. group the verbs by the nouns they relate
7. create Node objects such that:
  - Node's ids are the nouns and compound nouns from step 2.
  - Node's properties are the corresponding nouns and compound nouns adjectives from step 2.
  - Node's  types are the enriched classes from step 3.
  - carefully enrich the properties with biomedical contextual cues for enhancing semantics

8. create Relationship objects such that:
  - Relationship's types are the leading verbs derived from step 4.
  - Relationship's properties are the adverbs and compound adverbs from step 5. that are related to the Relationship's type.
  - Relationship's sources and targets are the Nodes whose ids are the nouns and compound nouns related by the Relationship's id verb.
  - carefully enrich the properties with biomedical contextual cues for enhancing semantics

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

10. return the result in a json object where the nodes and relationships are placed on a lists under `nodes` and `rels`, respectively; as in the step 10. example

the input text is:

{{ unstructured_text }}

11. double checking
  - if a relationship type be None, go back through to 1. and 4. 
  - if a relationship source cannot be None, go back to 6.
  - if a relationship target cannot be None, go back to 6.
  - if a relationship's source or target is None, remove the relationship
  - if rels be an empty list and nodes a non empty list, go back to 1. try it more carefully
  - if there is not a relationship for every two nodes, go back to 1.
  - all values must be alpha numeric strings

12. revisit empty relationships and make sure they are not empty. there should be at least one relationship per verb.
13. ensure that no keys nor values are lists. if that occurs, create new objects with one each.
14. ensure that the output is a valid JSON object.
15. enrich the properties of both nodes and relationship so as to improve semantics, contextualization and accuracy; maintain the biomedical domain
15. make the output JSON valid

the formatting instructions are: 

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

# slit_msg = """
# Given the text below:

# {{ unstructured_text }}

# 1. correct the punctuation
# 2. apply python's nltk sent_tokenize
# """

# list_parser = CommaSeparatedListOutputParser()
# prompt_splt = PromptTemplate(
#     template=slit_msg,
#     input_variables=["unstructured_text"],
#     # partial_variables={"format_instructions": list_parser.get_format_instructions()},
#     format="json",
#     template_format="jinja2",
#     validate_template=True,
# )


# punc_chain = {"unstructured_text": RunnablePassthrough()} | prompt_punctuation | llm | str_parser
# sent_chain = {"unstructured_text": RunnablePassthrough()} | prompt_sent | llm | lst_parser
sem_chain = {"unstructured_text": RunnablePassthrough()} | prompt_semantic | llm | json_parser
# sem_chain = {"unstructured_text": RunnablePassthrough()} | prompt_splt | llm | list_parser
# print(extract_msg_1)
texts = [
  "Coronaviruses are enveloped, single stranded, positive sense RNA viruses.",'Background: The key role of Vitamin D is to maintain an adequate calcium and phosphorus metabolism. Vitamin D plays an antagonistic role with the parathyroid hormone. 25 OH Vitamin D is the major circulating form and the best indicator to monitor Vitamin D levels.',
  'Methods: A cross-sectional study was conducted in 1339 individuals 18 years old. The main objective was to establish the nutritional status of Vitamin D and its association with PTH and ionized calcium levels. Other ob-jectives were to compare the levels of 25 OH Vitamin D based on sun exposure habits, and to identify the min-imum cut-off point for the levels of 25 OH Vitamin D that could give rise to a concomitant increase in PTH and ionized calcium levels.',
  'Results: 14.2% of participants presented Vitamin D deﬁciency, and 28.8% presented insufﬁciency; 89% of the participants with deﬁciency or insufﬁciency were exposed to sunlight <30 minutes per week. A value of 25 OH Vitamin D >30 ng/mL was associated with a more stable and “ﬂat” PTH value. The median of 25 OH Vit-D associated with hypercalcemia was <10 ng/mL.',
  'Conclusion: In Colombia, low 25 OH Vitamin D values are highly prevalent; this may be accounted for by poor sun-exposure habits and frequent use of sunscreen. Just as in other similar trials, the lower the levels of 25 OH Vit-D, the higher the effect on PTH and ionized calcium elevation.',
  'meostasis.HypovitaminosisDhasbeenassociatedwithcountlessboneand metabolic conditions, including rickets, osteomalacia, osteoporosis, increased falls risk, autoimmunity, cancer, diabetes mellitus, cardiovas-cular disease, etc. [11, 12]. Vit-D plays an antagonistic role with the parathyroid hormone (PTH), and hence “hypovitaminosis D” stimulates PTHsecretion,inducingtheremovalofcalciumfromthebone.25OHVit-D isthemajorcirculatingformandthebestindicatortomonitorVit-Dlevels [13, 14]. In 2011, the Institute of Medicine concluded that the adequate level of 25 OH Vit-D for optimum bone health was 20 ng/mL. This same year,theEndocrineSocietydeﬁnedthelevelsofVit-D(25OHVit-D)inthe population,indicatingthatlevels>30ng/mLare“optimal”,20–30ng/mL are “insufﬁcient”, and <20 ng/mL are “deﬁcient” [15, 16]. Current esti-mates indicate that around 15% of the world population is Vit-D deﬁcient or insufﬁcient and some of the factors involved are insufﬁcient sun exposure, poor Vit-D diets, use of sunscreens, old age, dark skin, inter alia * Corresponding author.',
  'E-mail address: hernandovargasuricoechea@gmail.com (V.-U. Hernando).',
  'Received 16 October 2019; Received in revised form 8 January 2020; Accepted 20 February 2020 2405-8440/© 2020 The Author(s). Published by Elsevier Ltd. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).',
  'V.-U. Hernando et al.',
  '[17, 18]. The main objective of this study was to establish the nutritional status of Vit-D (deﬁciency, insufﬁciency, and optimum values) and its as-sociation with PTH and ionized calcium levels in a population of the southeastern region in Colombia. Other objectives were: to do a socio-demographic characterization of the population, to compare the levels of 25 OH Vit-D based on socio-demographic characteristics and sun exposurehabits;toidentifyanychangesinPTHvaluesandionizedcalcium for different levels of 25 OH Vit-D; and to identify the minimum cut-off point for the levels of 25 OH Vit-D that could give rise to a concomitant increase in PTH and ionized calcium levels.',
  'protection factor –SPF-]. Body weight, height, and body mass index –BMI- [(kg/m2) classiﬁed as <18.5: low weight; between 18.5 and 24.9: normal; between 25 and 29.9: overweight; and 30: obese] were measured in all of the subjects. Then, fasting serum levels of 25 OH Vit-D, ionized calcium and PTH were also measured. The levels of 25 OH Vit-D (ng/mL) were measured using ELFA (Enzyme Linked Fluorescent Assay); and the results were quantitatively analyzed and classiﬁed based on The Endocrine Society (2011) guidelines. PTH (intact molecule) levels were measured through a solid phase chemiluminescent sequential immuno-assay, considering 11–67 pg/mL as a normal value. Ionized calcium levels were measured through an electrolyte analyzer, based on ion-selective electrode measurements, with normal values ranging from 1.13-1.51 mmol/L [19, 20, 21].',
  'Heliyon 6 (2020) e03479 of medical records and adhering to the rules of the Institutional Review Committee of Human Ethics (reference number: ID 5075). Universidad del Cauca, Popayan-Cauca-Colombia. Written informed consents were obtained from all participants.',
  "2.2. Statistical analysis The univariate analysis of qualitative variables was conducted using frequencies and percentages; the quantitative variables (considering that a non-normal distribution was identiﬁed in every case) were analyzed with nonparametric statistics [median, range, interquartile range (IQR)] using Shapiro Wilk. In the bivariate analysis, 25 OH Vit-D presented a non-normal distribution, and the Mann Withney U test was used to compare and contrast the 25 OH Vit-D values, against a number of var-iables including gender, sun exposure and use of sunscreen. The Kruskall Wallis test was used to compare the levels of 25 OH Vit-D in accordance with variables such as SEL, and BMI. Additionally, Spearman's correla-tion coefﬁcient was used to assess the relationship with quantitative variables (age, ionized calcium, PTH); this analysis considered the following scale: 0–0.25: poor or null correlation; 0.26–0.50: weak cor-relation; 0.51–0.75: moderate to strong correlation; 0.76–1.00: strong to perfect correlation. In order to identify any changes in the ionized cal-cium levels and PTH for the different levels of 25 OH Vit-D, summary measurements were presented at each level. These analyses were accompanied by tendency lines for each variable. ROC curves were used to identify the minimum cut-off point for the 25 OH Vit-D levels that resulted in a concomitant elevation of PTH or ionized calcium levels. A 95% conﬁdence interval was considered for all the analyses and the allowable α error was 0.05. Consequently, a p < 0.05 value was considered statistically signiﬁcant. The statistical analysis was conducted using SPSS Statistics V21.0.",
  '3. Results Out of 1457 individuals that met the criteria to participate in the study, 1339 participants were included (of the remaining 118 subjects, 49 refused to participate and 69 failed to properly complete the survey information, or the laboratory parameters).',
  '3.1. Socio-demographic characteristics, sun exposure habits, and distribution of 25 OH Vit-D, PTH, and ionized calcium levels Among the 1339 participants, 85.7% were females; 50% of the par-ticipants were between 44 and 68 years old (IQR:24); 63.7% claimed they had less than 30 min of sun exposure per week, and 57% used sunscreen <3 times per week. 56.8% of the participants were over-weight, and less than 15% had some level of obesity. The median 25 OH Vit-D was 32.3 ng/mL (IQR: 23.20). Moreover, 14.2% of the population presented Vit-D deﬁciency, 28.8% presented insufﬁciency, and 57% had optimum levels. The median PTH was 44.3 pg/mL, and 50% of the subjects exhibited levels between 29.4-73.9 pg/mL (IQR: 44.50). The ionized calcium median was 1.37 mmol/L, and 50% had values between 1.33-1.41 mmol/L (IQR: 0.08) (Table 1).',
  '3.2. Levels of 25 OH Vit-D, based on socio-demographic characteristics and sun exposure habits When comparing the levels of 25 OH Vit-D with the qualitative var-iables (using Mann-Withney U test) the levels of 25 OH Vit-D were found to be higher in women that received >30 min of sun exposure per week, and who claimed to use sunscreen <3 times/week (p ¼ 0.000). No sta-tistically signiﬁcant differences were found (using the Kruskall Wallis test) between the levels of 25 OH Vit-D and the SEL (p ¼ 0.482) (Table 2).',
  'V.-U. Hernando et al. Heliyon 6 (2020) e03479 population.',
  'Middle High Abbreviations: SEL: Socio-economic level; BMI: body mass index; IQR: interquartile range; PTH: parathyroid hormone; Vit-D: vitamin D.',
  'Levels of 25 OH Vit-D (ng/mL) Median Min.',
  'P value Abbreviations: Min: minimum, Max: maximum, SEL: socio-economic level, IQR: interquartile range, Vit-D: vitamin D. * Statistically signiﬁcant ﬁndings.',
  'V.-U. Hernando et al.',
  'inverse and signiﬁcant correlation (though scarce) between the levels of 25 OH Vit-D and age (R: 0.073; p ¼ 0.007). An inverse and scarce (non-signiﬁcant) correlation was also found between the levels of 25 OH Vit-D and BMI (-0.025; p ¼ 0.367) (Table 3).',
  '3.4. Correlation between the levels of 25 OH Vit-D, PTH and ionized calcium A low level of 25 OH Vit-D generated an increase in the PTH and ionized calcium values. The largest increase in the median PTH (from 57.7 to 89.75 pg/mL) occurred when changing 25 OH Vit-D from a range of 20–24.99 ng/mL to 15–19.99 ng/mL. In contrast, the highest increase in the ionized calcium median (from 1.39 to 1.48 mmol/L) occurred when changing 25 OH Vit-D from a range of 15–19.99 ng/mL to a range of 10–14.99ng/mL.A medianof 25 OH Vit-D of 33 ng/mLwasassociated with a PTH value of 35.2 pg/mL, and with an ionized calcium value of 1.13–1.15 mmol/L. Finally, a median of 25 OH Vit-D of 16.65 ng/mL was associated with PTH and ionized calcium values of 87.8 pg/mL and >1.51 mmol/L, respectively (Figures 1 and 2).',
  "3.5. Level of 25 OH Vit-D that better discriminates the normal PTH values The median value of 25 OH Vit-D that was associated with a normal range PTH value (11–67 pg/mL) was 35.5 ng/mL. In contrast, a level of <20 ng/mL was associated with PTH values > 67 pg/mL, while a value of >30 ng/mL was associated with a more stable and “ﬂat” PTH value (Figures 3 and 4). When evaluating the correlation between the levels of 25 OH Vit-D and PTH (Spearman's coefﬁcient), a weak and signiﬁcant correlation was found (-0.453; p ¼ 0.000); furthermore, a 25 OH Vit-D range between 25-29.99 ng/mL was found to generate a signiﬁcant change in the PTH median from 39.70 ng/mL to 42.30 pg/mL.",
  'Heliyon 6 (2020) e03479 pg/mL. The 25 OH Vit-D value that best discriminated this difference was 24.5 ng/mL (Figure 5).',
  "The regression analysis excluded 10 atypical extreme data (those that were 3 fold away from the IQR); 2 related to the 25 OH Vit-D variable, and 8 related to PTH. Upon removing these data, a non-normal distribu-tion persisted, generating a result of 0.908 (p: 0.00) and of 0.888 (p: 0.00) for 25 OH Vit-D and PTH, respectively, with the Shapiro Wilk test. Our evaluation of the correlation using the Spearman's, test identiﬁed an inverse scarce and statistically signiﬁcant correlation (-0.48, p: 0.000). A complementary analysis was conducted to explore the relationship be-tween the levels of 25 OH Vit-D and PTH values, through a non-lineal (logarithmic) regression. Similarly, the ANOVA analysis found that the logarithmic model was signiﬁcant to explain the dependent variable based on the independent variable, and veriﬁed that both B0 (159.433, p: 0.000) and B1 (-30.130, p: 0.000) were statistically different from 0.",
  "When checking the validity of the model adjustment, a determination coefﬁcient (R2) of 0.22 was identiﬁed (Figure 6). The residual assump-tions were checked to validate the model's predictive capacity, but the ﬁndings indicated that the residuals did not follow a normal distribution (P: 0.000), the residuals were not independent [checked through the run test - (p: 0.000) and the assumption of homoscedasticity was not met, since when increasing the adjustment, the variability increased]. All of this is accounted for by the weak correlation between the 2 variables. Notwithstanding the fact that the model showed a low predictive ability, the results obtained therefrom were used, which resulted in Y ¼ aþb*lnX, so that in the logarithmic regression equation the following values were identiﬁed: Y ¼ 159.433 þ (-30.130 (lnX)); a ¼ 159.433, b ¼ -30.130, and Y ¼ 67.01. Consequently, the 25 OH Vit-D value that originated a PTH value of >67 pg/mL was 21.49 ng/mL.",
  'Total (n ¼ 1339) Abbreviations: SEL: socio-economic level; BMI: body mass index; PTH: parathyroid hormone; Vit-D: vitamin D.',
  'V.-U. Hernando et al. Heliyon 6 (2020) e03479 V.-U. Hernando et al.',
  "mL was associated with ionized calcium values > 1.51 mmol/L (Figure 7). When assessing the correlation between 25 OH Vit-D and ionized calcium (using Spearman's coefﬁcient), an inverse and signiﬁcant correlation was identiﬁed (-2.258; p ¼ 0.000). A complementary ROC curves-based analysis was conducted, classifying individuals into 2 groups: the ﬁrst group was comprised of those individuals with ionized calcium levels 1.51 mmol/L; and the second group was comprised of individuals with levels of >1.51 mmol/L. The level of 25 OH Vit-D that best discriminated the difference between these 2 groups was 22.35 ng/ mL (Figure 8). Finally, a range of 25 OH Vit-D between 20-24.99 ng/mL led to a change in the ionized calcium median of 1.36–1.8 mmol/L (p ¼ 0.000).",
  '3.7. Correlation between PTH and ionized calcium A PTH median of 35.2 pg/mL was associated with normal ionized calcium levels (1.13–1.51 mmol/L); in contrast, a PTH median of 87.8 pg/mL was associated with an increased ionized calcium level >1.51 mmol/L. Furthermore, a PTH median of 3.6 pg/mL was associated with an ionized calcium level of <1.13 mmol/L (Figure 9).',
  "studies have shown that values of 25 OH Vit-D <16 ng/mL are associated with a signiﬁcant PTH elevation, and still others have shown a signiﬁcant change in the PTH value with a level of 25 OH Vit-D of 31.56 ng/mL [32, 33, 34]. This has led to the conclusion that the PTH plateau is reached with a 25 OH Vit-D level ranging between 8-44 ng/mL –but mostly be-tween 30-40 ng/mL- [35, 36]. Consequently, the plateau should be considered a “turning point”, which refers to the stimulation of PTH secretion when the levels of 25 OH Vit-D reach the limit to maintain the homeostasis of calcium. The variations in the cut-off point for 25 OH Vit-D among the various populations (resulting from the PTH plateau) may be at least partially accounted for by seasonal variations, de-mographic considerations, the way 25 OH Vit-D is measured, and ethnicity, inter alia. Notwithstanding the available evidence, the value of measuring the PTH concentration (to identify the optimum level of Vit-D) continues to be controversial and inconsistent; there is yet no threshold to deﬁne which is a “sufﬁcient” Vit-D status [37, 38]. Notwithstanding this situation, measuring 25 OH Vit-D continues to be the best indicator of the Vit-D status of the population because it does not depend on PTH levels and it directly reﬂects the body's Vit-D stores [39, 40, 41]. We found an inverse, weak and signiﬁcant correlation (probably due to the non-lineal nature of the association), among the levels of 25 OH Vit-D and PTH; and we were able to identify that the level of 25 OH Vit-D that originated a signiﬁcant change in PTH values was 24.5 ng/mL. However, the level of 25 OH Vit-D that resulted in a PTH value above the upper normal value (>67 pg/mL) was 21.49 ng/mL (this value will determine the PTH plateau for our population, since it was the “turning point” for the maximum PTH suppression). We therefore consider that a level of 25 OH Vit-D of <25 ng/mL may be adequate to determine Vit-D fortiﬁcation and/or supplementation. An additional ﬁnding was that low levels of 25 OH Vit-D resulted in signiﬁcant changes in ionized calcium values; for instance, the level of 25 OH Vit-D that generated a signiﬁcant change in the level of ionized calcium was 22.35 ng/mL, but the median of 25 OH Vit-D associated with an ionized calcium value above the upper normal level was <10 ng/mL. These results indicate –as expected-that measuring ionized calcium is useless for the early diagnosis of Vit-D insufﬁciency or deﬁciency (whilst PTH elevation seems to be an earlier V.-U. Hernando et al.",
  'Author contribution statement including the fact that the participants were followed at an intermediate complexity care center that cared for other various health conditions, different from metabolic bone disorders; therefore, this precludes the assumption that the results may be extrapolated to the rest of the pop-ulation.Moreover, other aspectsthat could interfere withthe values of 25 OH Vit-D were not considered (i.e., foods containing Vit-D, vegeta-rian-vegan lifestyles, physical activity, and skin pigmentation); addi-tionally, the measurement of 25 OH Vit-D was done through immunoassay(as opposedto chromatography, whichis a more consistent and accurate method). Having a history of osteomalacia, osteoporosis and/or fractures was not considered either, and hence we may not as-sume that the results obtained with regards to 25 OH Vit-D, PTH and ionized calcium values have any impact on the bone health of this pop-ulation. Furthermore, only one measurement of 25 OH Vit-D, PTH and ionized calcium was taken from the participants and so these results reﬂect the Vit-D status of the population at one point in time, and not necessarily over time.',
  'In contrast, the study has a number strengths such as the fact that the subjects did not receive any Vit-D, calcium or magnesium supplements (so the results may be considered independent from the use of those supplements); solar radiation in the geographical area is considered to be constant throughout the 12 months of the year (ruling out the potential “annual circadian rhythm” of 25 OH Vit-D which is typical of seasonal variations). Ionized calcium instead of total serum calcium was measured, so there was no need to correctfor albumin; moreover, ionized calcium values are a better reﬂection of calcium metabolism. Both, creatinine and glomerular ﬁltration rate were considered, and any in-dividuals with health conditions and/or use of medications that could eventually alter the levels of 25 OH Vit-D were excluded.',
  'In summary, the high prevalence of 25 OH Vit-D insufﬁciency and deﬁciency in this population is to a large extent the result of low sun exposure and of the frequent use of sunscreens (this does not preclude other confounding factors such as ethnicity, genetics, polymorphisms, skin pigmentation, physical activity, inter alia). A median of 25 OH Vit-D of 35.5 ng/mL was associated with a normal PTH range (11–67 pg/mL); whilst a range of 25 OH Vit-D between 20-24.99 ng/mL leads to a sig-niﬁcant increase in PTH and a level of 25 OH Vit-D <10 ng/mL results in a signiﬁcant increase in the level of ionized calcium. Finally, a level of 25 OH Vit-D of 16.65 ng/mL was associated with a signiﬁcant and concomitant increase in PTH and ionized calcium.',
  'Further studies in this geographical area should assess other aspects associated with the nutritional status of Vit-D, particularly among vulnerable population groups (children, adolescents, pregnant women, displaced populations, inter alia). Likewise, other sources of Vit-D, different from sun exposure, shall be assessed, as well as its impact on bone health,on varioustypes of cancers,and on chronic and autoimmune diseases.',
  '5. Conclusion In Colombia, where the amount of solar radiation is constant throughout the year, low 25 OH Vit-D values are highly prevalent; this may be accounted for by poor sun-exposure habits and frequent use of sunscreen. The PTH plateau was reached with a level of 25 OH Vit-D of 21.49 ng/mL, and the value of 25 OH Vit-D that led to a signiﬁcant change in PTH values was 24.5 ng/mL. The signiﬁcant increase in the levels of ionized calcium was experienced by individuals with low levels of 25 OH Vit-D, and concomitant PTH elevation, suggesting that the level of ionized calcium is not a useful marker for early detection of Vit-D deﬁciency/insufﬁciency. This suggests that in similar populations, a level of 25 OH Vit-D <25 ng/mL could be considered as the cut-off point to start Vit-D supplementation.',
  'H. Vargas-Uricoechea: Conceived and designed the experiments; Performed the experiments; Analyzed and interpreted the data; Contributed reagents, materials, analysis tools or data; Wrote the paper. M. Pinzon-Fernandez and V. Agredo: Performed the experiments; Contributed reagents, materials, analysis tools or data.',
  'A. Mera-Mamian: Analyzed and interpreted the data; Wrote the paper.',
  'Funding statement This research did not receive any speciﬁc grant from funding agencies in the public, commercial, or not-for-proﬁt sectors.',
  'Competing interest statement The authors declare no conﬂict of interest.',
  'Additional information No additional information is available for this paper.',
  'Table 1. Socio-demographic and clinical baseline characteristics, sun exposure habits, and distribution of 25 OH Vit-D levels, ionized calcium, and PTH in the study',
  'Table 2. Levels of 25 OH Vit-D according to socio-demographic variables, sun exposure habits, and use of sunscreen.',
  'Table 3. Levels of 25 OH Vit-D according to socio-demographic characteristics, habits, and ionized calcium and PTH values.',
  'Figure 1. Ionized calcium values and PTH, according to 25 OH Vit-D levels.',
  'Figure 2. Levels of 25 OH Vit and PTH, according to ionized calcium values.',
  'Figure 3. Median of 25 OH Vit-D and PTH values.',
  'Figure 4. Relationship between the levels of 25 OH Vit-D and PTH values.',
  'Figure 5. Level of 25 OH Vit-D and critical PTH value.',
  'Figure 6. Correlation between levels of 25 OH Vit-D and PTH values.',
  'Figure 7. Relationship between levels of 25 OH Vit-D and ionized calcium values.',
  'Figure 8. Levels of 25 OH Vit-D and critical ionized calcium value.',
  'Figure 9. Relationship between the PTH values and ionized calcium.'
]

for text in texts:
  print(text)
  sentences = nltk.sent_tokenize(text)
  print()
  for sentence in sentences:
    try:
      sem_data_1 = sem_chain.invoke({"unstructured_text": sentence})
      print(sem_data_1)
    except Exception as e:
      print(e, " on ", sentence)
  print()
  print()
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