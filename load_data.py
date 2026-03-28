#!/usr/bin/env python
# coding: utf-8

# 10.2. Clinical Relation Extraction Model Visualization with Neo4j


import os
import re
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product

from langchain.graphs.graph_document import (
    Node,
    Relationship,
    GraphDocument,
)


input_dir = Path("data__4096_0_plain")


input_dir = Path(f"graph")

assert input_dir.exists()

files = list(input_dir.rglob("*.csv"))


from neo4j import GraphDatabase

# Neo4j connection from environment variables
uri = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
username = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD")
if not password:
    raise EnvironmentError("NEO4J_PASSWORD environment variable is required.")


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

            # Use parameterized queries to prevent Cypher injection
            query = (
                f"MERGE (e1:{entity1} {{name: $chunk1}})"
                f"MERGE (e2:{entity2} {{name: $chunk2}})"
                f"MERGE (e1)-[:{relation} {{confidence: $confidence}}]->(e2)"
            )

            session.run(query, chunk1=chunk1, chunk2=chunk2, confidence=confidence)

    driver.close()


def insert_into_neo4j__prompted(data):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        for row in data[1:]:
            subject_name, subject_type, relationship, object_name, object_type = row

            # Use parameterized queries to prevent Cypher injection
            query = (
                f"MERGE (s:{subject_type} {{name: $subject_name}})"
                f"MERGE (o:{object_type} {{name: $object_name}})"
                f"MERGE (s)-[:{relationship}]->(o)"
            )

            session.run(query, subject_name=subject_name, object_name=object_name)

    driver.close()


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
    df["object_type"] = (
        df["object_type"].astype(str).apply(lambda x: x.replace(" ", "_"))
    )
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
