#!/usr/bin/env python
# coding: utf-8

# 10.2. Clinical Relation Extraction Model Visualization with Neo4j


import re
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product

# from utils import Neo4jConnection, get_relations_df, get_triples, update_data, add_ners_rels
from langchain.graphs.graph_document import (
    Node,
    Relationship,
    GraphDocument,
)


input_dir = Path("data__4096_0_plain")


input_dir = Path(f"graph")
# clinical_temp_dir = input_dir / "clinical_temp_events_re_pipeline"
# clinical_re_dir = input_dir / "clinical_re_pipeline"
# posology_dir = input_dir / "posology_relation_extraction_pipeline"

assert input_dir.exists()

files = list(input_dir.rglob("*.csv"))


from neo4j import GraphDatabase

# Neo4j connection details
uri = "bolt://localhost:7687"  # Update with your Neo4j server URI
username = "neo4j"  # Update with your Neo4j username
password = "password"  # Update with your Neo4j password


def insert_into_neo4j(data):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        for row in data[1:]:
            (
                relation,
                entity1,
                entity1_begin,
                entity1_end,
                chunk1,
                entity2,
                entity2_begin,
                entity2_end,
                chunk2,
                confidence,
            ) = row

            query = (
                f"MERGE (e1:{entity1} {{name: '{chunk1}'}})"
                f"MERGE (e2:{entity2} {{name: '{chunk2}'}})"
                f"MERGE (e1)-[:{relation} {{confidence: {confidence}}}]->(e2)"
            )

            session.run(query)

    driver.close()


def insert_into_neo4j__prompted(data):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        for row in data[1:]:
            subject_name, subject_type, relationship, object_name, object_type = row
            # print(row)
            query = (
                f"MERGE (s:{subject_type} {{name: '{subject_name}'}})"
                f"MERGE (o:{object_type} {{name: '{object_name}'}})"
                f"MERGE (s)-[:{relationship}]->(o)"
            )

            session.run(query)

    driver.close()


# Sample John Snow Labs data
# data = [
#     ["relation", "entity1", "entity1_begin", "entity1_end", "chunk1", "entity2", "entity2_begin", "entity2_end", "chunk2", "confidence"],
#     ["TrAP", "TREATMENT", 130, 137, "delivery", "PROBLEM", 142, 153, "the organism", 0.9999888],
#     ["TeRP", "TEST", 322, 342, "vitamin D 3absorption", "PROBLEM", 423, 438, "mainly dependent", 1.0],
#     ["TrAP", "TREATMENT", 652, 660, "vitamin D", "PROBLEM", 709, 715, "altered", 0.99999774],
#     ["TeRP", "TEST", 667, 681, "its metabolites", "PROBLEM", 709, 715, "altered", 1.0],
#     ["TeRP", "TREATMENT", 777, 832, "The digestion and absorption processes,on the other hand", "PROBLEM", 842, 856, "greatly altered", 1.0],
#     ["TeRP", "TREATMENT", 952, 968, "complex enzymatic", "PROBLEM", 977, 1001, "physicochemicalmechanisms", 0.8664989],
# ]

# insert_into_neo4j(data)

for f in files:
    df = pd.read_csv(f)

    df["subject_name"] = (
        df["subject_name"].astype(str).apply(lambda x: x.replace("-", "_"))
    )
    df["object_name"] = (
        df["object_name"].astype(str).apply(lambda x: x.replace("-", "_"))
    )
    df["relationship"] = (
        df["relationship"].astype(str).apply(lambda x: x.replace("-", "_"))
    )
    df["subject_type"] = (
        df["subject_type"].astype(str).apply(lambda x: x.replace(" ", "_"))
    )
    # df["object_name"] = df["object_name"].astype(str).apply(lambda x: x.replace("`", ""))
    df["object_type"] = (
        df["object_type"].astype(str).apply(lambda x: x.replace(" ", "_"))
    )
    # df["object_name"] = df["object_name"].astype(str).apply(lambda x: x.replace("'", ""))
    # df["relationship"] = df["relationship"].astype(str).apply(lambda x: x.replace("`", ""))
    # print(df)
    # df["chunk2"] = df["chunk2"].astype(str).apply(lambda x: x.replace("'", ""))
    df["subject_name"] = (
        df["subject_name"].astype(str).apply(lambda x: x.replace(" ", "_"))
    )
    df["object_name"] = (
        df["object_name"].astype(str).apply(lambda x: x.replace(" ", "_"))
    )
    df["relationship"] = (
        df["relationship"].astype(str).apply(lambda x: x.replace(" ", "_"))
    )
    data = df.values.tolist()

    insert_into_neo4j__prompted(data)
