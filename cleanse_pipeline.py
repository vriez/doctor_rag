import re
import copy
import spacy
import pandas as pd
import language_tool_python
from pathlib import Path
from docx import Document
from docx.shared import Cm
from itertools import product
from functools import partial
from multiprocessing import Pool

# from generate_rels import *

tool = language_tool_python.LanguageTool(
    "en-US"
)  # use a local server (automatically set up), language English

# Load the English language model
nlp = spacy.load("en_core_web_sm")


def starts_with_figure(row):
    pattern = r"^(Figure|Fig)\s*\d+[.:]"
    return re.match(pattern, row) is not None


def starts_with_table(row):
    pattern = r"^(Table|Tab)\s*\d+[.:]"
    return re.match(pattern, row) is not None


def is_finished(s):
    pattern = r"\.$"
    return bool(re.search(pattern, s))


def remove_consecutive_whitespaces(text):
    # Use regular expression to replace consecutive whitespaces with a single space
    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text


def remove_spaces_around_parentheses(text):
    # Remove spaces after opening parenthesis, bracket, or brace and before a word
    text = re.sub(r"(?<=[\(\[\{])\s+(?=\w)", "", text)
    # Remove spaces after a word and before closing parenthesis, bracket, or brace
    text = re.sub(r"(?<=\w)\s+(?=[\)\]\}])", "", text)
    return text


def select_text_smaller(document, font_size=8):
    # this is ok for we don't are not currently using the
    doc = copy.deepcopy(document)
    selected_text = []

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            # Check if the font size is smaller than font_size points
            if run.font.size and run.font.size.pt <= font_size:
                run._element.getparent().remove(run._element)
                # selected_text.append(run.text)
                # if run.text != "":
                #     print(run.text)
    return doc


def select_text_greater(document, font_size=13):
    # this is ok for we don't are not currently using the
    doc = copy.deepcopy(document)
    selected_text = []

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            # Check if the font size is smaller than font_size points
            if run.font.size and run.font.size.pt > font_size:
                run._element.getparent().remove(run._element)
                # selected_text.append(run.text)
                # if run.text != "":
                #     print(run.text)
    return doc


def merge_lowercase_with_hyphen(text):
    # Define a regex pattern to match lowercase strings containing hyphens
    pattern = r"\b[a-z]+-[a-z]+\b"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Replace each match with a merged string
    for match in matches:
        # Merge lowercase strings with hyphen
        merged_string = match.replace("-", "")
        # Replace the match with the merged string in the original text
        text = text.replace(match, merged_string)

    return text


def delete_images_and_tables(document):

    # doc = document.copy()
    doc = copy.deepcopy(document)
    images_to_delete = []
    tables_to_delete = []

    for idx, paragraph in enumerate(doc.paragraphs):
        # Check for images
        for run in paragraph.runs:
            if (
                run._element.tag
                == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r"
            ):
                for pic in run._element.iter(
                    "{http://schemas.openxmlformats.org/drawingml/2006/picture}pic"
                ):
                    images_to_delete.append((idx, pic))

    # Remove images
    for idx, pic in images_to_delete:
        doc.paragraphs[idx].clear()

    # Identify tables to delete
    tables_to_delete = []
    for idx, table in enumerate(doc.tables):
        tables_to_delete.append(idx)

    # Remove tables
    for idx in reversed(tables_to_delete):
        doc._element.body.remove(doc.tables[idx]._element)

    return doc


def text_to_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def correct_morphology(text):
    matches = tool.check(text)
    # select the matches to be applid to the text
    for m in matches:
        if m.ruleIssueType == "misspelling":
            try:
                m.replacements = [m.replacements[1]]
            except IndexError:
                continue
        else:
            matches.remove(m)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text


def pipeline(document):

    doc = delete_images_and_tables(document)
    # doc = select_text_smaller(doc)
    # doc = select_text_greater(doc)
    doc = document

    attachments = []
    show = False
    data = []
    keywords = ["abstract", "introduction", "summary"]
    # is_present = any(keyword in p.text.lower() for keyword in keywords)
    last_data = ""
    for p in doc.paragraphs:
        if starts_with_table(p.text) or starts_with_figure(p.text):
            attachments.append(p.text)
            continue

        if p.text.upper() == "REFERENCES":
            show = False
            continue

        if show is False:

            if any(keyword in p.text.lower() for keyword in keywords):
                show = True
                continue

        if p.text.isnumeric():
            continue
        if p.text.upper() == p.text:
            continue

        if p.text != "" and show:
            if is_finished(p.text):
                last_data += " " + p.text
                # print(last_data)
                # continuous_text = merge_lowercase_with_hyphen(last_data)
                # morph_text = correct_morphology(continuous_text)
                last_data = remove_consecutive_whitespaces(last_data).strip()
                # clean_text = remove_spaces_around_parentheses(clean_text)
                # clean_text = clean_text.strip()
                # clean_text = re.sub(r"\n+", " ", clean_text)
                # clean_sentences = clean_text.strip().replace(". ", ".<|>").split("<|>")

                # clean_sentences = text_to_sentences(clean_text)

                # data.extend(clean_sentences)
                # print(last_data)
                # print()
                data.append(last_data)

                last_data = ""
            else:
                last_data = p.text.replace("-\n", "")
            # continue
        # print(last_data)
    data.extend(attachments)
    return data


input_folder = Path("dataset_docx_ocr")

input_files = input_folder.glob("*.docx")

# input_files = [Path("dataset_docx_ocr/0aefb1158b98bc12005c0f74c9e9987e.docx")]

# dataset = []
text_rels_pairs = []
for f in input_files:
    document = Document(f)
    data = pipeline(document)

    # for d in data:
    #     # rels = ner_extract(d)
    #     row = {
    #         "descriptor": f.stem,
    #         "text": d,
    #         # "ner": rels
    #     }
    #     text_rels_pairs.append(row)

    data = " ".join(data)

    #     prompt = f"""
    # given the following context:

    # {data}

    # perform the procedure below:

    # 1. correct the punctuation, spelling and typos.
    # 2. camel-case exclusively all entities, e.g., noun-phrases, nouns, concepts, percentages, p-values, ranges, dosages, concentrations, techniques, methodologies, time periods, time ranges, age ranges, authors, scientific references and other clinical trial related parameters; consider the biomedical context, following the guidelines:
    #     a- an entity must represent a specific semantic meaning
    #     b- an entity must not start with a preposition
    #     c- an entity should not be too long
    #     d- a camel-cased entity must contain at most 3 nouns
    #     e- all adjectives must be present in the output

    # 3. snake-case exclusively all phrasal verbs, prepositional verbs, verb-particle constructions, inseparable phrasal verbs, idiomatic verbs, idiomatic expressions, two consecutive verbs and modal verbs, following the guidelines:
    #     a- verbs must only be concatenated with other surrounding verbs, copulas and prepositions
    #     b- adverbs must not be concatenated with a noun
    #     c- a snake-cased verbs must contain no nouns nor articles

    # have you overlooked any of the 2.a, 2.b, 2.c or 2.d, 2.e guidelines? make all necessary corrections.
    # have you overlooked any of the 3.a, 3.b or 3.c guidelines? make all necessary corrections.

    # no words can be removed from the corrected context
    # no words can be added to the corrected context

    # ensure all phrasal verbs are snake-cased
    # ensure all prepositional verbs are snake-cased
    # ensure all verb-particle constructions are snake-cased
    # ensure all inseparable phrasal verbs are snake-cased
    # ensure all idiomatic verbs are snake-cased
    # ensure all idiomatic expressions are snake-cased
    # ensure all consecutive verbs are snake-cased
    # ensure all modal verbs are snake-cased

    # ensure all noun-phrases are camel-cased
    # ensure all nouns are camel-cased
    # ensure all concepts are camel-cased
    # ensure all percentages are camel-cased
    # ensure all p-values are camel-cased
    # ensure all ranges are camel-cased
    # ensure all dosages are camel-cased
    # ensure all concentrations are camel-cased
    # ensure all techniques are camel-cased
    # ensure all methodologies are camel-cased
    # ensure all time periods are camel-cased
    # ensure all time ranges are camel-cased
    # ensure all age ranges are camel-cased
    # ensure all authors are camel-cased
    # ensure all scientific references are camel-cased

    #     """
    # print(name, len(data))
    with open(f"docx_2_text/{f.stem}.txt", "w") as f:
        f.write(data)
    # f_name = f.stem
    # data = list(product([f_name], data))
    # dataset.extend(data)
    # print(pd.DataFrame(text_rels_pairs))

# df = pd.DataFrame(text_rels_pairs)
# print(df)
# df.to_csv("doc_clean_stripped_f_abstract.csv", index=None)


# def process_document(f):
#     document = Document(f)
#     data = pipeline(document)  # Assuming you have defined your pipeline function
#     f_name = f.stem
#     data = list(product([f_name], data))
#     return data

# # Define the number of processes
# num_processes = 8  # Adjust as needed

# # Create a Pool of processes
# with Pool(num_processes) as pool:
#     # Map the process_document function to each input file
#     results = pool.map(process_document, input_files)

# # Flatten the results list
# dataset = [item for sublist in results for item in sublist]

# # Create a DataFrame from the dataset
# df = pd.DataFrame(dataset)

# # Save the DataFrame to CSV
# df.to_csv("doc_clean_parallel_data.csv", index=None)
