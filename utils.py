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

def dataset_overlaap(df, chunk_size, overlap):
    nodes = []
    files = df.groupby("fname")

    for f_name, f_content in files:
        chunks = []
        start_index = 0  # Initialize start index for chunks

        while start_index < len(f_content):
            # Calculate end index for the chunk considering the chunk_size
            end_index = min(start_index + chunk_size, len(f_content))

            # Extract chunk's content and create metadata
            chunk_df = f_content.iloc[start_index:end_index]
            metadata = {
                "source": f_name,
                "block_size": chunk_size,
                "start": chunk_df.index[0],
                "end": chunk_df.index[-1],
            }
            
            # Combine texts within the chunk
            text = ' '.join(chunk_df['text'].str.lower())

            # Create a Document object and add to nodes
            node = Document(text=text.strip(), metadata=metadata)
            nodes.append(node)

            # Update start_index for the next chunk considering overlap
            start_index = end_index - overlap

            # Prevent infinite loop at the end of a file by ensuring progress
            if start_index + overlap >= len(f_content):
                break
        break
    return nodes

def dataset_overlap(df, chunk_size, overlap):
    # overlap += 1
    nodes = []
    files = df.groupby("fname")

    for f_name, f_content in files:
        chunk = ""
        start = None  # track the start from the group index
        # print(f_name, f_content.shape, f_content.index)

        doc_start = f_content.index[0]
        doc_end = f_content.index[-1]
        print("File start at: ", f_name, doc_start, doc_end)
        block_start = doc_start
        block_end = block_start

        block_id = 0
        # print("index: ", f_content.index)
        # break
        df_i = iter(enumerate(f_content.iterrows()))
        i = 0
        
        tail = False
        wrap_up = False
        print("-----------> ", i)
        while True:
            print("K: ", block_end, block_end - 1 == doc_end, chunk == "")
            if i > len(f_content):
                print("Whoa")
                break
        # for _ in range(doc_start, doc_end+1):
        # for id, (i, row) in enumerate(f_content.iterrows()):
            # print("i is: ", i)
            # try:
            try:
                row = f_content.loc[block_end]
                content = row.text.lower()
            except:
                print("M: ", )
                tail = True
                # break
            # print("F: ", i, row)
            # id, (i, row) = next(df_i)
            # print("> ", id, i, row)
            # except KeyError:
            #     content = ""
            #     wrap_up = True
            #     # print(id, i, content)
            size = len(content)
            

            if (len(chunk) + size >= chunk_size) ^ tail:
                # if tail:
                #     print("~~~")

                # block_end = i
                # if (id == doc_end):
                #     print("ends before")

                block_id += 1

                # Handle multi-row chunk creation
                metadata = {
                    "source": f_name,
                    "block_size": chunk_size,
                    "size": len(chunk),
                    "start": block_start,
                    "end": block_end,
                }

                # print(metadata)
                
                df_content = df.iloc[block_start : block_end, :]
                df_content = df_content[df_content["fname"] == f_name]["text"]
                
                # print("==> ", df_content)
                
                text = "\n".join(df_content)
                node = Document(text=text.strip(), metadata=metadata)
                nodes.append(node)
                chunk = ""
                
                # start = i + overlap + 1  # Update start for the next chunk
                print(f"Block {block_id} starts at: {block_start} - {block_end} - {doc_end} | {i} ++ {len(text)}", metadata)
                block_start = block_end - overlap
                tail = False

            else:
                # print("len == ", len(chunk))
                # Accumulate text for multi-row chunks
                chunk += " " + content
                # if i == doc_end+1:
                #     print("== ", block_end, i, doc_end)
                #     block_end = doc_end
                # else:
                block_end += 1
                i += 1
            
            if block_end - 1 == doc_end and chunk == "":
                print("Ka: ", block_end, doc_end, block_end - 1 == doc_end, chunk == "")
                break
        # else:
        #     print("V: ")
            
            
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
        print(doc_text)
        docs.append(doc)
        break
    return docs
