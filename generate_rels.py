from transformers import pipeline
import pandas as pd

triplet_extractor = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large",
    tokenizer="Babelscape/rebel-large",
)

# We need to use the tokenizer manually since we need special tokens.


# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def ner_extract(text):

    extracted_text = triplet_extractor.tokenizer.batch_decode(
        [
            triplet_extractor(text, return_tensors=True, return_text=False)[0][
                "generated_token_ids"
            ]
        ]
    )

    extracted_triplets = extract_triplets(extracted_text[0])

    return extracted_triplets


# df = pd.read_csv("doc_clean_data.csv")

# df["ner"] = df["1"].apply(ner_extract)
# for d in data:
#     ers = ner_extract(d)
#     print(ers)
