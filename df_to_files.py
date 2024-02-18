import pandas as pd
from pathlib import Path

# df = pd.read_csv("doc_clean_data_sentences_idx.csv")

# docs = df.groupby("descriptor")

input_files = Path("docx_2_text").glob("*.txt")

for i_f in input_files:
    # print(doc)
    name = i_f.stem
    # data = " ".join(doc["text"].values)
    with open(i_f, "r") as f:
        data = f.read()

    prompt = f"""
given the following context:

{data}

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

    """
    print(name, len(data))
    with open(f"prep_prompts/{name}.txt", "w") as f:
        f.write(prompt)
