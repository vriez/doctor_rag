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
    assert len(data) > 0, i_f
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

perform the following procedure deterministically:

1. correct the punctuation, spelling and typos.
2. camel-case exclusively all entities, e.g., noun-phrases, nouns, concepts, percentages, p-values, ranges, dosages, concentrations, techniques, latin terms and phrases, methodologies, time periods, time ranges, age ranges, authors, scientific references and other clinical trial related parameters; consider the biomedical context, following the guidelines:
    a- represent a semantic meaning
    b- not start with a preposition
    c- not be overly long (contain at most 3 nouns)
    e- all adjectives must be present in the output
    f- acronyms and initialisms related to diseases, viruses, and other scientific terms should be treated as single entities without separators (e.g., "COVID-19" becomes "Covid19", "SARS-CoV-2" becomes "SarsCov2",  "25(OH'D or 25(OH)D or 25 (OH' D" become "25OHD", )
    g- numerical parts of entities should be directly attached to the preceding text without spaces or hyphens
    h- apply camel-casing to compound nouns and adjectives to form a single entity without spaces, ensuring the first letter of each subsequent word is capitalized (e.g., "Vitamin D" becomes "VitaminD", "fat soluble" becomes "fatSoluble", "13.2±5.0 ng/mL" becomes "13.2±5.0NgML").

3. snake-case exclusively all phrasal verbs, prepositional verbs, verb-particle constructions, inseparable phrasal verbs, idiomatic verbs, idiomatic expressions, two consecutive verbs and modal verbs, following the guidelines:
    a- verbs should only be concatenated with other surrounding verbs, copulas and prepositions
    b- adverbs must not be concatenated with a noun
    c- ensure that snake-cased verbs contain no nouns nor articles
    d- verb phrases should maintain readability and natural language flow
    e- avoid concatenating verbs in a manner that creates unwieldy or unnatural phrases. 
    f- or complex verb phrases where snake-casing would impair readability, maintain standard spelling and spacing.
    g- adverbs must not be concatenated with a noun
    h- use snake-casing for phrasal verbs by connecting all parts of the verb (including prepositions and adverbs) with underscores (e.g., "is thought of" becomes "is_thought_of", "provided by" becomes "provided_by")
    i- apply snake-casing to modal verbs (e.g., could, should, would) when they are immediately followed by another verb, and optionally a preposition or adverb, to form a continuous phrase without altering the natural flow of the sentence. This includes constructions where the modal verb is part of a verb phrase that implies an action or state (e.g., "could be used" becomes "could_be_used").
    j- avoid concatenating verbs in a manner that creates overly long or confusing phrases. In cases where a verb phrase becomes too complex or long due to snake-casing, evaluate if the phrase can be simplified or if standard spelling should be retained for clarity


4. Numerical Values and Units: Combine numerical values and their corresponding units into a single entity by capitalizing the first letter of each significant part of the unit after the numerical value. This approach is applicable for various dosages, enhancing readability and consistency.
    For instance, "13.8 pg/mL" should be written as "13.8PgML", and "50 mg" as "50Mg".

5. Ranges and Intervals: Clearly denote ranges or intervals by attaching numerical values directly to their units and applying camel-casing without spaces. Use a dash to signify the range, and ensure clarity by including the unit after each numerical value if necessary.
    As an example, "3-7 days/week" could become "3–7DaysPerWeek", maintaining clear communication of the range.

6. Complex Units: When dealing with units that comprise multiple components (e.g., concentration, frequency), deconstruct the unit into its fundamental elements, apply camel-casing, and then recombine them into a cohesive, readable format.
    For example, "units per liter per day" should be transformed into "UnitsPerLiterPerDay".

7. Frequency and Duration: Specify frequency (how often something occurs) or duration (how long something lasts) by directly attaching numbers to units without spaces, applying camel-casing, and capitalizing each part to clearly separate words.
    For instance, "twice daily" becomes "TwiceDaily", and "4 times a week" becomes "4TimesAWeek".

8. General Principles:
    Clarity Over Brevity: While it's important to strive for conciseness, clarity should always be the priority. Ensure that the transformed dosage or measurement is easily understandable.
    Semantic Integrity: Preserve the original meaning of the dosage or measurement. The camel-cased entity should convey all necessary information clearly and accurately.
    Consistency: Apply these guidelines uniformly across all documents and texts to ensure that all dosages and related measurements are consistently formatted, making them easier for readers to comprehend and interpret.

have you overlooked any of the 2.a - 2.h guidelines? make all necessary corrections.
have you overlooked any of the 3.a - 3.j guidelines? make all necessary corrections.

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
