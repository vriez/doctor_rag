import re
import hashlib
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# df_1 = pd.read_excel("vitamind_insufficiency_deficiency__sunscreen.xlsx", sheet_name="savedrecs", skiprows=10)
# df_2 = pd.read_excel("vitamind_insufficiency_deficiency__covid19.xlsx", sheet_name="savedrecs", skiprows=10)
# df_3 = pd.read_excel("sunscreen_covid19.xlsx", sheet_name="savedrecs", skiprows=10)
# df_4 = pd.read_excel("vitamind__covid19_severity_recovery_infection.xlsx", sheet_name="savedrecs", skiprows=10)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


df_1 = pd.read_excel("busca_1.xlsx", sheet_name="savedrecs", skiprows=10)
df_2 = pd.read_excel("busca_2.xlsx", sheet_name="savedrecs", skiprows=10)
df_3 = pd.read_excel("busca_3.xlsx", sheet_name="savedrecs", skiprows=10)
df_4 = pd.read_excel("busca_4.xlsx", sheet_name="savedrecs", skiprows=10)

df_1["busca"] = 1
df_2["busca"] = 2
df_3["busca"] = 3
df_4["busca"] = 4

def calculate_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, 'rb') as file:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


cc = pd.concat([df_1, df_2, df_3, df_4])

catalog_df = pd.read_csv("dataset.csv"
                      )
input_dir = Path("busca")
output_dir = Path("busca__lower")

wos = cc[["Title", "Source Title", "DOI", "busca"]]


merged_df = pd.merge(catalog_df, wos, left_on='title', right_on='Title', how='inner')

data = []

print("Drep: ", merged_df)

for index, row in merged_df.iterrows():
    # print("row: ", row)
    hash = row.md5sum
    file_path = Path(row.file_name)

    # pages = PyPDFLoader(str(file_path)).load()
    hash = calculate_md5(file_path)
    size = file_path.stat().st_size
    merged_df.at[index, 'size'] = size
    # page_count = 0
    # char_count = 0
    
    # for page in pages:
    #     page_count += 1
    #     char_count += len(page.page_content)
    
    # print(file_path, page_count, char_count)
    # merged_df.at[index, 'page_count'] = page_count
    # merged_df.at[index, 'char_count'] = char_count




# for _, (t, s, doi, b) in cc[["Title", "Source Title", "DOI", "busca"]].iterrows():
#     fs = str(output_dir / t.replace("/", "").replace(' ', '_').replace(":", "").replace('-', '').replace('–', '').lower()[:222]) + f"__busca_{b}.pdf"
#     fs = fs.replace("___busca", "__busca")
#     clean = re.compile('<.*?>')
#     fs = re.sub(clean, '', fs)
#     fs = Path(fs)

#     try:
#         pages = PyPDFLoader(str(fs)).load()
#         hash = calculate_md5(fs)
#         size = fs.stat().st_size
#     except ValueError:
#         pages = []
#         hash = None
#         size = 0


#     dt = {
#         "file_name": Path(f"busca__final/{hash}.pdf"),  # Fix the closing parenthesis
#         "md5sum": hash,
#         "page_count": 0,
#         "char_count": 0,
#         "size": size,
#         "doi": doi,
#         "title": t,
#         "source": s,
#     }

#     for page in pages:
#         if fs.exists():
#             dt["page_count"] += 1
#             dt["char_count"] += len(page.page_content)

#     data.append(dt)

# df = pd.DataFrame(data)

# df.to_csv("")

