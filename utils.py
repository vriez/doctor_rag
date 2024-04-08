import google.generativeai
import time
from neo4j import exceptions
from llama_index.core.schema import Document


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
    overlap += 1
    nodes = []
    files = df.groupby("fname")

    for f_name, f_content in files:
        chunk = ""
        start = None  # track the start from the group index
        # print(f_content.shape)
        for id, (i, row) in enumerate(f_content.iterrows()):
            content = row.text.lower()
            # print(id, i)
            size = len(content)
            if size > chunk_size:
                # Handle single-row chunks directly
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": size,
                    "start": i - overlap,
                    "end": i - overlap,
                }
                # print(metadata)
                df_content = df.iloc[i + 1 - overlap : i + 1, :]
                df_content = df_content[df_content["fname"] == f_name]["text"]
                # print("!=> ", df_content)
                text = "\n".join(df_content)
                node = Document(text=text.strip(), metadata=metadata)
                nodes.append(node)
                start = (
                    i + overlap + 1
                )  # Update start for potential subsequent multi-row chunks
                continue

            elif len(chunk) + size > chunk_size:
                # Handle multi-row chunk creation
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": len(chunk),
                    "start": start - overlap,
                    "end": i - overlap,
                }
                # print(metadata)
                df_content = df.iloc[start - overlap : i + overlap, :]
                df_content = df_content[df_content["fname"] == f_name]["text"]
                # print("==> ", df_content)
                text = "\n".join(df_content)
                node = Document(text=text.strip(), metadata=metadata)
                nodes.append(node)
                chunk = ""
                start = i + overlap + 1  # Update start for the next chunk
                continue

            elif id > f_content.shape[0] - overlap:
                # Accumulate text for multi-row chunks
                chunk += " " + content

                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": len(chunk),
                    "start": start - overlap,
                    "end": i - overlap,
                }
                # start -= overlap
                # print("-> ", " ".join(df.iloc[start + 1 - overlap:i, 2]))
                # continue
                # print(id, i, f_content.shape[0])

                df_content = df.iloc[start - overlap - 1 : i + overlap, :]
                df_content = df_content[df_content["fname"] == f_name]["text"]
                # print("--> ", metadata, df_content)
                text = "\n".join(df_content)

                node = Document(text=text.strip(), metadata=metadata)
                nodes.append(node)
                start = i + overlap + 1
            else:
                # Accumulate text for multi-row chunks
                chunk += " " + content
                start = start or i
                continue
        # break
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
        print(doc_text)
        docs.append(doc)
        break
    return docs
