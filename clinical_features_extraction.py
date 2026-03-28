#!/usr/bin/env python
# coding: utf-8

# 10.2. Clinical Relation Extraction Model Visualization with Neo4j


import re
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from utils import Neo4jConnection, get_relations_df, get_triples
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from summary import *
from sparknlp.base import LightPipeline


# uri = "bolt://localhost:7687"
# pwd = "password"
# user = "neo4j"
# conn = Neo4jConnection(uri=uri, user=user, pwd=pwd)

def infer(pipeline, data):
    empty_data = spark.createDataFrame([[""]]).toDF("text")
    model = pipeline.fit(empty_data)
    lmodel = LightPipeline(model)
    annotations = lmodel.fullAnnotate(data)
    res_df = get_relations_df(annotations)
    return res_df

loader = PyPDFDirectoryLoader("busca__final")
docs = loader.load()

CHUNK_SIZE = [512, 1024, 2048, 4096]
CHUNK_OVERLAP = [0, 24, 56]

catalog_df = pd.read_csv("catalog.csv")

for chunk_size, chunk_overlap in product(CHUNK_SIZE, CHUNK_OVERLAP):
    # print("M: ", chunk_size, chunk_overlap)

    output_dir = Path(f"cleaned_data__{chunk_size}_{chunk_overlap}")
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_temp_dir = output_dir / "clinical_temp_events_re_pipeline"
    clinical_temp_dir.mkdir(parents=True, exist_ok=True)

    clinical_re_dir = output_dir / "clinical_re_pipeline"
    clinical_re_dir.mkdir(parents=True, exist_ok=True)

    posology_dir = output_dir / "posology_relation_extraction_pipeline"
    posology_dir.mkdir(parents=True, exist_ok=True)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)

    offset = 0
    chunks = chunks[offset:]


    for i, chunk in tqdm(enumerate(chunks, offset), total=len(chunks)):
        
        text = chunk.page_content
        file_path = chunk.metadata.get("source")
        file_name = Path(file_path).stem

        # value_of_B = catalog_df.loc[catalog_df['md5sum'] == hash, 'B'].values[0]
        doc_text = chunk.page_content.replace('-\n', '')
        # doc_text = re.sub(r'(?<!\.)\n', ' ', doc_text)

        if not Path(clinical_temp_dir / f"{i}_{file_name}.csv").exists():
            events_df = infer(clinical_temp_events_re_pipeline, text)
            events_df.to_csv(clinical_temp_dir / f"{i}_{file_name}.csv", index=None)

        if not Path(clinical_re_dir / f"{i}_{file_name}.csv").exists():
            clinical_re_df = infer(clinical_re_pipeline, text)
            clinical_re_df.confidence = clinical_re_df.confidence.astype(float)
            # clinical_re_df = clinical_re_df[clinical_re_df.relation != "O"]
            # print("clinical_re_pipeline: ", clinical_re_df)
            clinical_re_df.to_csv(clinical_re_dir / f"{i}_{file_name}.csv", index=None)

        if not Path(posology_dir / f"{i}_{file_name}.csv").exists():
            posology_df = infer(posology_relation_extraction_pipeline, text)
            # print("posology_relation_extraction_pipeline: ", posology_df)
            posology_df.to_csv(posology_dir / f"{i}_{file_name}.csv", index=None)

        # graph_df = infer(graph_extraction_pipeline, text)

        # # df = spark.createDataFrame(
        # df = pd.DataFrame({"id": [0], "text" : [doc_text] })
        # sc = spark.sparkContext

        # # Create a PySpark DataFrame with a predefined schema
        # schema = StructType([StructField("id", IntegerType(), True), StructField("text", StringType(), True)])
        # rdd = sc.parallelize(df.to_records())
        # spark_df = spark.createDataFrame(rdd, schema=schema)
        # result = graph_extraction_pipeline.fit(spark_df).transform(spark_df)
        # graph_df = get_graph_result(result)
        # print("graph_extraction_pipeline: ", graph_df)
        # graph_df.to_csv(f"js_data/{i}__graph_relation_extraction_pipeline.csv", index=None)

        # # add_ners_rels(rel_df)

output_dir = Path(f"cleaned_data__plain")
output_dir.mkdir(parents=True, exist_ok=True)


for i, chunk in tqdm(enumerate(docs, offset), total=len(chunks)):
    
    text = chunk.page_content
    file_path = chunk.metadata.get("source")
    file_name = Path(file_path).stem

    # value_of_B = catalog_df.loc[catalog_df['md5sum'] == hash, 'B'].values[0]
    doc_text = chunk.page_content.replace('-\n', '')
    # doc_text = re.sub(r'(?<!\.)\n', ' ', doc_text)

    events_df = infer(clinical_temp_events_re_pipeline, text)
    events_df.to_csv(clinical_temp_dir / f"{i}_{file_name}.csv", index=None)

    clinical_re_df = infer(clinical_re_pipeline, text)
    clinical_re_df.confidence = clinical_re_df.confidence.astype(float)
    # clinical_re_df = clinical_re_df[clinical_re_df.relation != "O"]
    # print("clinical_re_pipeline: ", clinical_re_df)
    clinical_re_df.to_csv(clinical_re_dir / f"{i}_{file_name}.csv", index=None)

    posology_df = infer(posology_relation_extraction_pipeline, text)
    # print("posology_relation_extraction_pipeline: ", posology_df)
    posology_df.to_csv(posology_dir / f"{i}_{file_name}.csv", index=None)
