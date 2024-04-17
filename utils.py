import time
import hashlib
import google.generativeai
from neo4j import exceptions
from llama_index.core.schema import Document


def hash_string(string):
    """Hashes a string using a cryptographic hash function (SHA256).

    Args:
        string: The string to hash.

    Returns:
        A string representation of the hash value.
    """
    # Encode the string as bytes (utf-8 encoding)
    string_encoded = string.encode("utf-8")
    # Create a SHA256 hash object
    h = hashlib.sha256()
    # Update the hash object with the string bytes
    h.update(string_encoded)
    # Return the hex digest of the hash
    return h.hexdigest()


def dataset(df, chunk_size):

    nodes = []
    files = df.groupby("fname")

    for f_name, f_content in files:
        chunk = ""
        start = None  # track the start from the group index

        for i, row in f_content.iterrows():
            content = row.text.lower()
            size = len(content)
            if size > chunk_size:
                # Handle single-row chunks directly
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": size,
                    "start": i + 1,
                    "end": i + 1,
                }

                node = Document(text=content.strip(), metadata=metadata)
                nodes.append(node)
                start = i  # Update start for potential subsequent multi-row chunks
                continue

            elif len(chunk) + size > chunk_size:
                # Handle multi-row chunk creation
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": len(chunk),
                    "start": start + 1,
                    "end": i,
                }

                node = Document(text=chunk.strip(), metadata=metadata)
                nodes.append(node)
                chunk = ""
                start = i  # Update start for the next chunk
                continue

            else:
                # Accumulate text for multi-row chunks
                chunk += " " + content
                start = start or i
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": len(chunk),
                    "start": start + 1,
                    "end": i,
                }
                # continue
            text = " ".join(df.iloc[start + 1 : i, 2])
            node = Document(text=text.strip(), metadata=metadata)
            nodes.append(node)

    return nodes


def dataset_overlap(df, chunk_size, overlap):
    nodes = []
    hashes = []
    files = df.groupby("fname")  # Assuming 'fname' is the filename column

    for f_name, f_content in files:
        chunk = ""
        block_start = f_content.index[0]
        block_end = block_start
        block_id = 0

        while block_end <= f_content.index[-1]:
            if block_end in f_content.index:
                row = f_content.loc[block_end]
                content = row.text.lower()
                new_chunk_size = len(chunk) + len(content)

                if new_chunk_size >= chunk_size:
                    # Create metadata for the current chunk
                    metadata = {
                        "source": f_name,
                        "block_size": chunk_size,
                        "size": len(chunk),
                        "start": block_start,
                        "end": block_end,
                    }

                    # Collect the text for the current chunk
                    df_content = df.loc[block_start:block_end]
                    df_content = df_content[df_content["fname"] == f_name]["text"]
                    text = "\n".join(df_content)

                    hashed_value = hash_string(text)
                    if hashed_value in hashes:
                        print("duplicated at 1: ", new_chunk_size, chunk_size)
                        # break
                    else:
                        hashes.append(hashed_value)
                        nodes.append(Document(text=text.strip(), metadata=metadata))

                    # Update the start for the next chunk, considering the overlap
                    block_start = max(block_start, block_end - overlap)
                    chunk = ""

                # Add current content to chunk
                chunk += " " + content if chunk else content

            # Move to the next row
            block_end += 1

            # Special handling for the last chunk in the group
            if block_end > f_content.index[-1] and chunk:
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": len(chunk),
                    "start": block_start,
                    "end": f_content.index[-1],
                }
                df_content = df.loc[block_start : f_content.index[-1]]
                df_content = df_content[df_content["fname"] == f_name]["text"]
                text = "\n".join(df_content)
                hashed_value = hash_string(text)
                if hashed_value in hashes:
                    print("duplicated at 2: ", new_chunk_size, chunk_size)
                    # break
                else:
                    hashes.append(hashed_value)
                    # nodes.append({"text": text.strip(), "metadata": metadata})
                    nodes.append(Document(text=text.strip(), metadata=metadata))
                break

    return nodes


def dataset_whole(df):

    docs = []
    files = sorted(df.groupby("fname"))

    for f_name, f_content in files:
        f_content = f_content[f_content["fname"] == f_name]["text"]
        doc_text = " ".join(f_content)
        size = len(doc_text)
        metadata = {
            "source": f_name,
            "size": size,
            "start": f_content.index[0],
            "end": f_content.index[-1],
        }
        doc = Document(text=doc_text.strip(), metadata=metadata)
        # print(doc_text)
        docs.append(doc)
        # break
    return docs
