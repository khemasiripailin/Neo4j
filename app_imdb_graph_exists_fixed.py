# -*- coding: utf-8 -*-
"""
Streamlit app สำหรับงาน Neo4j Graph Data Science
- Network visualizer จาก edges_rows.csv + favorites_rows.csv
- Centrality visualization: Betweenness, Bridges, Closeness, Degree, Eigenvector, PageRank
- Ex.3 K-means community detection on edges_rows.csv / edges+favorites
- Ex.4 K-means community detection on karate_club
- Ex.5 K-means community detection on imdb โดยใช้ gds.graph.load_imdb()

สำคัญ:
- Centrality และ K-means คำนวณด้วย Neo4j GDS จริง
- Python/Streamlit ใช้สำหรับอ่านไฟล์ ส่งข้อมูลเข้า Neo4j และ visualize เท่านั้น
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from pyvis.network import Network

st.set_page_config(
    page_title="Neo4j GDS Network Visualizer",
    page_icon="🕸️",
    layout="wide",
)

DEFAULT_DATABASE = "neo4j"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    graph_name: str
    node_label: str
    rel_type: str
    title: str


VISUAL_CONFIG = DatasetConfig(
    key="visual_edges_favorites",
    graph_name="visualEdgesFavoritesGraph",
    node_label="VisualNode",
    rel_type="VISUAL_REL",
    title="Network visualizer",
)

CENTRALITY_CONFIG = DatasetConfig(
    key="centrality_edges_favorites",
    graph_name="centralityEdgesFavoritesGraph",
    node_label="CentralityNode",
    rel_type="CENTRALITY_REL",
    title="Centrality on edges_rows.csv + favorites_rows.csv",
)

KMEANS_EDGES_CONFIG = DatasetConfig(
    key="kmeans_edges_favorites",
    graph_name="kmeansEdgesFavoritesGraph",
    node_label="KMeansEdgesNode",
    rel_type="KMEANS_EDGES_REL",
    title="Ex.3 K-means on edges_rows.csv",
)

KARATE_CONFIG = DatasetConfig(
    key="karate_club",
    graph_name="karateKMeansGraph",
    node_label="KarateNode",
    rel_type="KARATE_REL",
    title="Ex.4 K-means on karate_club",
)


# -----------------------------
# Neo4j helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_driver(uri: str, username: str, password: str):
    return GraphDatabase.driver(uri, auth=(username, password))


def run_query(driver, database: str, query: str, **params) -> List[Dict[str, Any]]:
    with driver.session(database=database) as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


def check_gds(driver, database: str) -> str:
    rows = run_query(driver, database, "RETURN gds.version() AS version")
    return rows[0]["version"]


def safe_drop_graph(driver, database: str, graph_name: str) -> None:
    try:
        run_query(
            driver,
            database,
            "CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName",
            graph_name=graph_name,
        )
    except Exception:
        pass


def clear_dataset_nodes(driver, database: str, config: DatasetConfig) -> None:
    run_query(driver, database, f"MATCH (n:{config.node_label}) DETACH DELETE n")


def import_edges_to_neo4j(driver, database: str, config: DatasetConfig, edges_df: pd.DataFrame) -> Dict[str, int]:
    """นำ edge list เข้า Neo4j โดยใช้ label/relationship ตาม config"""
    safe_drop_graph(driver, database, config.graph_name)
    clear_dataset_nodes(driver, database, config)

    work = edges_df.copy()
    if "source_type" not in work.columns:
        work["source_type"] = "Person"
    if "target_type" not in work.columns:
        work["target_type"] = "Person"
    work["source_type"] = work["source_type"].fillna("Person").astype(str)
    work["target_type"] = work["target_type"].fillna("Person").astype(str)
    work["relation"] = work["relation"].fillna("knows").astype(str)

    rows = work[["source", "target", "relation", "source_type", "target_type"]].drop_duplicates().to_dict("records")

    query = f"""
    UNWIND $rows AS row
    MERGE (a:{config.node_label} {{name: row.source}})
    SET a.display_name = row.source,
        a.node_type = row.source_type
    MERGE (b:{config.node_label} {{name: row.target}})
    SET b.display_name = row.target,
        b.node_type = row.target_type
    MERGE (a)-[r:{config.rel_type} {{pair_id: row.source + '|' + row.target + '|' + row.relation}}]->(b)
    SET r.relation = row.relation
    """
    run_query(driver, database, query, rows=rows)

    counts = run_query(
        driver,
        database,
        f"""
        MATCH (n:{config.node_label})
        WITH count(n) AS nodes
        MATCH ()-[r:{config.rel_type}]->()
        RETURN nodes, count(r) AS edges
        """,
    )
    return counts[0]


def project_graph(driver, database: str, config: DatasetConfig) -> Dict[str, Any]:
    """สร้าง graph projection แบบ undirected"""
    safe_drop_graph(driver, database, config.graph_name)
    query = f"""
    CALL gds.graph.project(
        $graph_name,
        '{config.node_label}',
        {{
            {config.rel_type}: {{
                orientation: 'UNDIRECTED'
            }}
        }}
    )
    YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """
    return run_query(driver, database, query, graph_name=config.graph_name)[0]


# -----------------------------
# Data helpers
# -----------------------------
def read_default_or_upload(uploaded_file, default_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(default_path)


def _first_column_as_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    obj = df[col_name]
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj


def normalize_edges_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลงไฟล์ edges_rows.csv เป็น columns:
    source, target, relation, source_type, target_type

    รองรับไฟล์ที่มีชื่อ column ซ้ำ เช่น source, source, target, target
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_set = set(df.columns)

    if {"person_a", "person_b"}.issubset(col_set):
        source_series = _first_column_as_series(df, "person_a")
        target_series = _first_column_as_series(df, "person_b")
    elif {"source", "target"}.issubset(col_set):
        source_series = _first_column_as_series(df, "source")
        target_series = _first_column_as_series(df, "target")
    else:
        raise ValueError("ไฟล์ edges_rows.csv ต้องมีคอลัมน์ person_a/person_b หรือ source/target")

    if "relation" in col_set:
        relation_series = _first_column_as_series(df, "relation")
    else:
        relation_series = pd.Series(["knows"] * len(df), index=df.index)

    clean = pd.DataFrame(
        {
            "source": source_series.astype(str).str.strip(),
            "target": target_series.astype(str).str.strip(),
            "relation": relation_series.fillna("knows").astype(str).str.strip(),
        }
    )
    clean["relation"] = clean["relation"].replace({"": "knows", "nan": "knows", "None": "knows"})
    clean = clean[(clean["source"] != "") & (clean["target"] != "")]
    clean = clean[clean["source"] != clean["target"]]

    # unique undirected person-person edge
    clean["u"] = clean[["source", "target"]].min(axis=1)
    clean["v"] = clean[["source", "target"]].max(axis=1)
    clean = clean.drop_duplicates(subset=["u", "v", "relation"])

    result = pd.DataFrame(
        {
            "source": clean["u"].values,
            "target": clean["v"].values,
            "relation": clean["relation"].values,
            "source_type": "Person",
            "target_type": "Person",
        }
    ).sort_values(["source", "target", "relation"]).reset_index(drop=True)

    return result


def normalize_favorites_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    แปลงไฟล์ favorites_rows.csv เป็น edge list แบบ Person --likes--> Food
    รองรับชื่อคอลัมน์ person_name/food หรือ person/favorite/food_name
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    person_col = None
    for cand in ["person_name", "person", "name", "source"]:
        if cand in lower_map:
            person_col = lower_map[cand]
            break

    food_col = None
    for cand in ["food", "favorite", "favorite_food", "target"]:
        if cand in lower_map:
            food_col = lower_map[cand]
            break

    if person_col is None or food_col is None:
        raise ValueError("ไฟล์ favorites_rows.csv ต้องมีคอลัมน์ person_name และ food")

    person_series = _first_column_as_series(df, person_col)
    food_series = _first_column_as_series(df, food_col)

    clean = pd.DataFrame(
        {
            "source": person_series.astype(str).str.strip(),
            "target": food_series.astype(str).str.strip(),
            "relation": "likes",
            "source_type": "Person",
            "target_type": "Food",
        }
    )
    clean = clean[(clean["source"] != "") & (clean["target"] != "")]
    clean = clean[(clean["source"].str.lower() != "nan") & (clean["target"].str.lower() != "nan")]
    clean = clean.drop_duplicates(subset=["source", "target", "relation"])
    return clean.sort_values(["source", "target"]).reset_index(drop=True)


def build_graph_edges(person_edges: pd.DataFrame, favorite_edges: pd.DataFrame, include_favorites: bool) -> pd.DataFrame:
    if include_favorites and not favorite_edges.empty:
        graph_edges = pd.concat([person_edges, favorite_edges], ignore_index=True)
    else:
        graph_edges = person_edges.copy()
    graph_edges = graph_edges[["source", "target", "relation", "source_type", "target_type"]].drop_duplicates().reset_index(drop=True)
    return graph_edges


def get_node_table(edges_df: pd.DataFrame) -> pd.DataFrame:
    left = edges_df[["source", "source_type"]].rename(columns={"source": "node", "source_type": "node_type"})
    right = edges_df[["target", "target_type"]].rename(columns={"target": "node", "target_type": "node_type"})
    nodes = pd.concat([left, right], ignore_index=True).drop_duplicates(subset=["node"])
    return nodes.sort_values(["node_type", "node"]).reset_index(drop=True)


def get_basic_network_stats(edges_df: pd.DataFrame) -> Dict[str, Any]:
    nodes = get_node_table(edges_df)
    G = nx.Graph()
    for row in nodes.itertuples(index=False):
        G.add_node(row.node, node_type=row.node_type)
    for row in edges_df.itertuples(index=False):
        G.add_edge(row.source, row.target, relation=row.relation)
    components = list(nx.connected_components(G))
    diameter = "N/A (disconnected)"
    if nx.is_connected(G) and len(G) > 1:
        diameter = nx.diameter(G)
    largest_component = max((len(c) for c in components), default=0)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(components),
        "largest_component": largest_component,
        "diameter": diameter,
    }


def make_karate_edges() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """สร้าง Zachary's Karate Club graph เป็น edge list เพื่อส่งเข้า Neo4j GDS"""
    G = nx.karate_club_graph()
    nodes_df = pd.DataFrame(
        [{"node": str(node), "club": data.get("club", "unknown")} for node, data in G.nodes(data=True)]
    )
    edges_df = pd.DataFrame(
        [
            {
                "source": str(u),
                "target": str(v),
                "relation": "member_connection",
                "source_type": "KarateMember",
                "target_type": "KarateMember",
            }
            for u, v in G.edges()
        ]
    )
    return edges_df, nodes_df


def add_karate_club_property(driver, database: str, nodes_df: pd.DataFrame) -> None:
    rows = nodes_df.to_dict("records")
    query = """
    UNWIND $rows AS row
    MATCH (n:KarateNode {name: row.node})
    SET n.original_club = row.club
    """
    run_query(driver, database, query, rows=rows)


# -----------------------------
# IMDb sample graph by GDS Python Client
# -----------------------------
def unwrap_gds_graph(obj):
    """รองรับกรณี GDS client บางเวอร์ชันคืนค่าเป็น Graph หรือ tuple/list ที่มี Graph อยู่ตัวแรก"""
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        return obj[0]
    return obj


def get_client_graph_name(graph_obj) -> str:
    """ดึงชื่อ graph จาก Graph object ของ graphdatascience client ให้ได้มากที่สุด"""
    for attr in ["name", "graph_name", "graphName"]:
        if hasattr(graph_obj, attr):
            value = getattr(graph_obj, attr)
            try:
                value = value() if callable(value) else value
                if value:
                    return str(value)
            except Exception:
                pass
    # fallback: ใช้ string representation
    return str(graph_obj)


def load_imdb_with_gds_client(uri: str, username: str, password: str, database: str):
    """
    ทำตามโจทย์อาจารย์: G = gds.graph.load_imdb()
    ต้องใช้ package graphdatascience ซึ่งเป็น Neo4j GDS Python Client
    """
    try:
        from graphdatascience import GraphDataScience
    except Exception as exc:
        raise RuntimeError(
            "ยังไม่มี package graphdatascience ให้ติดตั้งด้วย: py -m pip install graphdatascience"
        ) from exc

    gds = GraphDataScience(uri, auth=(username, password), database=database)

    # ถ้ากด Run ซ้ำ Graph Projection ชื่อ imdb จะค้างอยู่ใน Neo4j GDS Catalog
    # จึง drop ก่อนโหลดใหม่ เพื่อกัน error: Graph 'imdb' already exists
    try:
        gds.run_cypher("CALL gds.graph.drop('imdb', false) YIELD graphName RETURN graphName")
    except Exception:
        pass

    loaded = gds.graph.load_imdb()
    graph_obj = unwrap_gds_graph(loaded)
    graph_name = get_client_graph_name(graph_obj)
    return gds, graph_obj, graph_name


def run_imdb_kmeans(driver, database: str, graph_name: str, k: int, random_seed: int) -> pd.DataFrame:
    """K-means สำหรับ IMDb graph โดยใช้ชื่อ/label ที่อาจต่างจาก graph คน/อาหาร"""
    query = """
    CALL gds.kmeans.stream(
        $graph_name,
        {
            nodeProperty: 'embedding',
            k: $k,
            randomSeed: $random_seed,
            initialSampler: 'kmeans++',
            numberOfRestarts: 5,
            maxIterations: 30,
            computeSilhouette: true
        }
    )
    YIELD nodeId, communityId, distanceFromCentroid, silhouette
    WITH gds.util.asNode(nodeId) AS n, nodeId, communityId, distanceFromCentroid, silhouette
    RETURN nodeId,
           coalesce(n.name, n.title, n.display_name, n.id, toString(nodeId)) AS node,
           labels(n) AS labels,
           coalesce(head(labels(n)), 'IMDbNode') AS node_type,
           communityId,
           distanceFromCentroid,
           silhouette
    ORDER BY communityId, node
    """
    try:
        return pd.DataFrame(run_query(driver, database, query, graph_name=graph_name, k=k, random_seed=random_seed))
    except Neo4jError:
        query2 = """
        CALL gds.kmeans.stream(
            $graph_name,
            {nodeProperty: 'embedding', k: $k, randomSeed: $random_seed}
        )
        YIELD nodeId, communityId, distanceFromCentroid
        WITH gds.util.asNode(nodeId) AS n, nodeId, communityId, distanceFromCentroid
        RETURN nodeId,
               coalesce(n.name, n.title, n.display_name, n.id, toString(nodeId)) AS node,
               labels(n) AS labels,
               coalesce(head(labels(n)), 'IMDbNode') AS node_type,
               communityId,
               distanceFromCentroid,
               0.0 AS silhouette
        ORDER BY communityId, node
        """
        return pd.DataFrame(run_query(driver, database, query2, graph_name=graph_name, k=k, random_seed=random_seed))


def get_imdb_edges_from_projection(driver, database: str, graph_name: str, limit: int = 600) -> pd.DataFrame:
    """ดึง edge จาก graph projection ของ IMDb เพื่อเอามาวาดกราฟใน Streamlit"""
    query = """
    CALL gds.graph.relationships.stream($graph_name)
    YIELD sourceNodeId, targetNodeId, relationshipType
    WITH gds.util.asNode(sourceNodeId) AS s,
         gds.util.asNode(targetNodeId) AS t,
         sourceNodeId, targetNodeId, relationshipType
    RETURN coalesce(s.name, s.title, s.display_name, s.id, toString(sourceNodeId)) AS source,
           coalesce(t.name, t.title, t.display_name, t.id, toString(targetNodeId)) AS target,
           relationshipType AS relation,
           coalesce(head(labels(s)), 'IMDbNode') AS source_type,
           coalesce(head(labels(t)), 'IMDbNode') AS target_type
    LIMIT $limit
    """
    try:
        rows = run_query(driver, database, query, graph_name=graph_name, limit=int(limit))
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame(columns=["source", "target", "relation", "source_type", "target_type"])


def run_imdb_pipeline(
    driver,
    database: str,
    uri: str,
    username: str,
    password: str,
    k: int,
    embedding_dim: int,
    random_seed: int,
    edge_limit: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """โหลด IMDb ด้วย gds.graph.load_imdb(), สร้าง embedding, แล้วรัน K-means ด้วย Neo4j GDS"""
    _, graph_obj, graph_name = load_imdb_with_gds_client(uri, username, password, database)
    # graph_name ต้องเป็นชื่อ graph projection ที่ GDS ใช้งานได้
    embedding_info = run_fastrp_embedding(driver, database, graph_name, embedding_dim, random_seed)
    results_df = run_imdb_kmeans(driver, database, graph_name, k, random_seed)
    edges_df = get_imdb_edges_from_projection(driver, database, graph_name, edge_limit)

    graph_info = {"graphName": graph_name}
    for method_name in ["node_count", "relationship_count", "nodeCount", "relationshipCount"]:
        if hasattr(graph_obj, method_name):
            try:
                value = getattr(graph_obj, method_name)
                graph_info[method_name] = value() if callable(value) else value
            except Exception:
                pass
    return results_df, edges_df, graph_info, embedding_info


# -----------------------------
# Centrality by Neo4j GDS
# -----------------------------
def gds_metric(driver, database: str, graph_name: str, procedure: str, metric_name: str) -> pd.DataFrame:
    query = f"""
    CALL {procedure}($graph_name)
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS name, score AS {metric_name}
    ORDER BY {metric_name} DESC
    """
    return pd.DataFrame(run_query(driver, database, query, graph_name=graph_name))


def get_all_nodes(driver, database: str, label: str) -> pd.DataFrame:
    rows = run_query(driver, database, f"MATCH (p:{label}) RETURN p.name AS name, p.node_type AS node_type ORDER BY name")
    if not rows:
        return pd.DataFrame(columns=["name", "node_type"])
    return pd.DataFrame(rows)


def gds_bridges(driver, database: str, graph_name: str) -> pd.DataFrame:
    """หา bridge edges ด้วย Neo4j GDS โดยรองรับ output ต่างกันในแต่ละ version"""
    query = """
    CALL gds.bridges.stream($graph_name)
    YIELD from, to
    RETURN gds.util.asNode(from).name AS source,
           gds.util.asNode(to).name AS target
    ORDER BY source, target
    """
    try:
        df = pd.DataFrame(run_query(driver, database, query, graph_name=graph_name))
        if df.empty:
            return pd.DataFrame(columns=["source", "target"])
        return df[["source", "target"]]
    except Neo4jError:
        raw = pd.DataFrame(
            run_query(
                driver,
                database,
                "CALL gds.bridges.stream($graph_name) YIELD * RETURN *",
                graph_name=graph_name,
            )
        )
        if raw.empty:
            return pd.DataFrame(columns=["source", "target"])
        possible_from = [c for c in raw.columns if c.lower() in {"from", "fromnodeid", "sourcenodeid"}]
        possible_to = [c for c in raw.columns if c.lower() in {"to", "tonodeid", "targetnodeid"}]
        if not possible_from or not possible_to:
            return pd.DataFrame(columns=["source", "target"])
        from_col, to_col = possible_from[0], possible_to[0]
        node_ids = sorted(set(raw[from_col].tolist() + raw[to_col].tolist()))
        name_rows = run_query(
            driver,
            database,
            "UNWIND $node_ids AS nodeId RETURN nodeId, gds.util.asNode(nodeId).name AS name",
            node_ids=node_ids,
        )
        id_to_name = {r["nodeId"]: r["name"] for r in name_rows}
        return pd.DataFrame({"source": raw[from_col].map(id_to_name), "target": raw[to_col].map(id_to_name)})


def calculate_centrality(driver, database: str, config: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = get_all_nodes(driver, database, config.node_label)
    metric_specs = [
        ("gds.betweenness.stream", "betweenness"),
        ("gds.closeness.stream", "closeness"),
        ("gds.degree.stream", "degree"),
        ("gds.eigenvector.stream", "eigenvector"),
        ("gds.pageRank.stream", "pagerank"),
    ]
    result = nodes.copy()
    for procedure, metric_name in metric_specs:
        metric_df = gds_metric(driver, database, config.graph_name, procedure, metric_name)
        result = result.merge(metric_df, on="name", how="left")

    result = result.fillna({"node_type": "Unknown", "betweenness": 0, "closeness": 0, "degree": 0, "eigenvector": 0, "pagerank": 0})

    bridges_df = gds_bridges(driver, database, config.graph_name)
    bridge_count: Dict[str, int] = {}
    for _, row in bridges_df.iterrows():
        bridge_count[str(row["source"])] = bridge_count.get(str(row["source"]), 0) + 1
        bridge_count[str(row["target"])] = bridge_count.get(str(row["target"]), 0) + 1
    result["bridge_count"] = result["name"].map(bridge_count).fillna(0).astype(int)

    return result.sort_values("betweenness", ascending=False).reset_index(drop=True), bridges_df


# -----------------------------
# K-means by Neo4j GDS
# -----------------------------
def run_fastrp_embedding(driver, database: str, graph_name: str, embedding_dim: int, random_seed: int) -> Dict[str, Any]:
    query = """
    CALL gds.fastRP.mutate(
        $graph_name,
        {
            embeddingDimension: $embedding_dim,
            randomSeed: $random_seed,
            mutateProperty: 'embedding'
        }
    )
    YIELD nodePropertiesWritten
    RETURN nodePropertiesWritten
    """
    return run_query(driver, database, query, graph_name=graph_name, embedding_dim=embedding_dim, random_seed=random_seed)[0]


def run_kmeans(driver, database: str, graph_name: str, k: int, random_seed: int) -> pd.DataFrame:
    query = """
    CALL gds.kmeans.stream(
        $graph_name,
        {
            nodeProperty: 'embedding',
            k: $k,
            randomSeed: $random_seed,
            initialSampler: 'kmeans++',
            numberOfRestarts: 5,
            maxIterations: 30,
            computeSilhouette: true
        }
    )
    YIELD nodeId, communityId, distanceFromCentroid, silhouette
    RETURN gds.util.asNode(nodeId).name AS node,
           gds.util.asNode(nodeId).node_type AS node_type,
           communityId,
           distanceFromCentroid,
           silhouette
    ORDER BY communityId, node
    """
    try:
        return pd.DataFrame(run_query(driver, database, query, graph_name=graph_name, k=k, random_seed=random_seed))
    except Neo4jError:
        query2 = """
        CALL gds.kmeans.stream(
            $graph_name,
            {nodeProperty: 'embedding', k: $k, randomSeed: $random_seed}
        )
        YIELD nodeId, communityId, distanceFromCentroid
        RETURN gds.util.asNode(nodeId).name AS node,
               gds.util.asNode(nodeId).node_type AS node_type,
               communityId,
               distanceFromCentroid,
               0.0 AS silhouette
        ORDER BY communityId, node
        """
        return pd.DataFrame(run_query(driver, database, query2, graph_name=graph_name, k=k, random_seed=random_seed))


def write_communities_to_neo4j(driver, database: str, config: DatasetConfig, results_df: pd.DataFrame) -> None:
    rows = results_df[["node", "communityId", "distanceFromCentroid", "silhouette"]].to_dict("records")
    query = f"""
    UNWIND $rows AS row
    MATCH (n:{config.node_label} {{name: row.node}})
    SET n.kmeans_community = row.communityId,
        n.kmeans_distance = row.distanceFromCentroid,
        n.kmeans_silhouette = row.silhouette
    """
    run_query(driver, database, query, rows=rows)


def run_kmeans_pipeline(
    driver,
    database: str,
    config: DatasetConfig,
    edges_df: pd.DataFrame,
    k: int,
    embedding_dim: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    import_counts = import_edges_to_neo4j(driver, database, config, edges_df)
    projection_info = project_graph(driver, database, config)
    embedding_info = run_fastrp_embedding(driver, database, config.graph_name, embedding_dim, random_seed)
    results_df = run_kmeans(driver, database, config.graph_name, k, random_seed)
    write_communities_to_neo4j(driver, database, config, results_df)
    return results_df, import_counts, projection_info, embedding_info


# -----------------------------
# Visualization helpers
# -----------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def interpolate_color(value: float, min_value: float, max_value: float, low: str = "#deebf7", high: str = "#08306b") -> str:
    if max_value <= min_value:
        ratio = 0.15
    else:
        ratio = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
    lr, lg, lb = hex_to_rgb(low)
    hr, hg, hb = hex_to_rgb(high)
    return rgb_to_hex((int(lr + (hr - lr) * ratio), int(lg + (hg - lg) * ratio), int(lb + (hb - lb) * ratio)))


def color_for_community(community_id: int, max_community: int) -> str:
    # ไล่สีจากม่วง -> น้ำเงิน -> เขียว -> เหลือง เพื่อให้เห็น community เป็น gradient
    palette = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]
    if max_community <= 0:
        return palette[2]
    ratio = max(0.0, min(1.0, community_id / max_community))
    idx_float = ratio * (len(palette) - 1)
    idx = int(idx_float)
    if idx >= len(palette) - 1:
        return palette[-1]
    frac = idx_float - idx
    c1 = hex_to_rgb(palette[idx])
    c2 = hex_to_rgb(palette[idx + 1])
    return rgb_to_hex(tuple(int(c1[i] + (c2[i] - c1[i]) * frac) for i in range(3)))


def get_degree_map(edges_df: pd.DataFrame) -> Dict[str, int]:
    degree: Dict[str, int] = {}
    for row in edges_df.itertuples(index=False):
        degree[row.source] = degree.get(row.source, 0) + 1
        degree[row.target] = degree.get(row.target, 0) + 1
    return degree


def bridge_set_from_df(bridges_df: pd.DataFrame) -> set:
    bridge_set = set()
    if bridges_df.empty:
        return bridge_set
    for _, row in bridges_df.iterrows():
        bridge_set.add(tuple(sorted((str(row["source"]), str(row["target"])))) )
    return bridge_set


def add_html_legend(html: str, legend_body: str) -> str:
    legend = f"""
    <div style="position:absolute; top:14px; right:14px; z-index:9999; background:white; border:1px solid #ddd; border-radius:12px; padding:12px 14px; font-family:Arial; font-size:13px; box-shadow:0 2px 8px rgba(0,0,0,.15); min-width:240px;">
      <div style="font-weight:700; margin-bottom:8px;">Legend</div>
      {legend_body}
    </div>
    """
    return html.replace("<body>", f"<body>{legend}").replace("©", "(c)")


def make_visual_network_html(edges_df: pd.DataFrame) -> str:
    degree = get_degree_map(edges_df)
    nodes_df = get_node_table(edges_df)
    min_degree = min(degree.values()) if degree else 0
    max_degree = max(degree.values()) if degree else 1

    net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-35000, central_gravity=0.2, spring_length=140, spring_strength=0.02, damping=0.08)

    for row in nodes_df.itertuples(index=False):
        node = str(row.node)
        node_type = str(row.node_type)
        d = degree.get(node, 0)
        if node_type == "Food":
            color = "#ffb703"
            shape = "diamond"
            size = 20 + d * 2
        else:
            color = interpolate_color(d, min_degree, max_degree, low="#deebf7", high="#08306b")
            shape = "dot"
            size = 16 + d * 2
        title = f"<b>{node}</b><br>ประเภท: {node_type}<br>Degree: {d}"
        net.add_node(node, label=node, title=title, color=color, shape=shape, size=size)

    for edge in edges_df.itertuples(index=False):
        if edge.relation == "likes":
            net.add_edge(edge.source, edge.target, label="likes", title="likes", color="#ffb703", width=2, dashes=True)
        else:
            net.add_edge(edge.source, edge.target, label=str(edge.relation), title=str(edge.relation), color="#9e9e9e", width=1)

    net.set_options('{"nodes":{"font":{"size":16}},"edges":{"smooth":false,"font":{"size":10,"align":"middle"}},"interaction":{"hover":true,"navigationButtons":true,"keyboard":true},"physics":{"enabled":true,"stabilization":{"iterations":250}}}')
    legend_body = """
      <div style="margin-bottom:8px;">สี node คน = ไล่สีตามจำนวน connection</div>
      <div style="height:12px; background:linear-gradient(90deg,#deebf7,#08306b); border-radius:6px; margin-bottom:4px;"></div>
      <div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-bottom:8px;"><span>Degree ต่ำ</span><span>Degree สูง</span></div>
      <div><span style="color:#ffb703; font-size:18px;">◆</span> Food node จาก favorites_rows.csv</div>
      <div><span style="border-top:2px dashed #ffb703; display:inline-block; width:34px; vertical-align:middle;"></span> likes</div>
      <div><span style="border-top:2px solid #9e9e9e; display:inline-block; width:34px; vertical-align:middle;"></span> knows</div>
    """
    return add_html_legend(net.generate_html(notebook=False), legend_body)


def make_centrality_network_html(edges_df: pd.DataFrame, centrality_df: pd.DataFrame, bridges_df: pd.DataFrame) -> str:
    metrics = centrality_df.set_index("name").to_dict("index") if not centrality_df.empty else {}
    bridge_edges = bridge_set_from_df(bridges_df)
    max_bet = float(centrality_df["betweenness"].max()) if not centrality_df.empty else 0.0
    min_bet = float(centrality_df["betweenness"].min()) if not centrality_df.empty else 0.0

    net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=170, spring_strength=0.08)

    for name, row in metrics.items():
        bet = float(row.get("betweenness", 0))
        node_type = str(row.get("node_type", "Person"))
        size = 14 + (48 * bet / max_bet if max_bet > 0 else 0)
        if node_type == "Food":
            color = "#ffb703"
            shape = "diamond"
            size = max(size, 24)
        else:
            color = interpolate_color(bet, min_bet, max_bet, low="#deebf7", high="#08306b")
            shape = "dot"
        title = (
            f"<b>{name}</b><br>"
            f"ประเภท: {node_type}<br>"
            f"Betweenness: {row.get('betweenness', 0):.6f}<br>"
            f"Closeness: {row.get('closeness', 0):.6f}<br>"
            f"Degree: {row.get('degree', 0):.6f}<br>"
            f"Eigenvector: {row.get('eigenvector', 0):.6f}<br>"
            f"PageRank: {row.get('pagerank', 0):.6f}<br>"
            f"Bridge count: {int(row.get('bridge_count', 0))}"
        )
        net.add_node(name, label=name, title=title, color=color, shape=shape, size=size)

    for row in edges_df.itertuples(index=False):
        a, b = str(row.source), str(row.target)
        is_bridge = tuple(sorted((a, b))) in bridge_edges
        if is_bridge:
            net.add_edge(a, b, color="#d62728", width=4, dashes=True, title="Bridge edge", label="bridge")
        elif str(row.relation) == "likes":
            net.add_edge(a, b, color="#ffb703", width=2, dashes=True, title="likes", label="likes")
        else:
            net.add_edge(a, b, color="#9e9e9e", width=1, title=str(row.relation), label=str(row.relation))

    net.set_options('{"nodes":{"font":{"size":16}},"edges":{"smooth":false,"font":{"size":10,"align":"middle"}},"interaction":{"hover":true,"navigationButtons":true,"keyboard":true},"physics":{"enabled":true,"stabilization":{"iterations":250}}}')
    legend_body = """
      <div style="margin-bottom:8px;">สี node คน = Betweenness Centrality</div>
      <div style="height:12px; background:linear-gradient(90deg,#deebf7,#08306b); border-radius:6px; margin-bottom:4px;"></div>
      <div style="display:flex; justify-content:space-between; font-size:11px; color:#555; margin-bottom:8px;"><span>ต่ำ</span><span>สูง</span></div>
      <div><span style="color:#ffb703; font-size:18px;">◆</span> Food node</div>
      <div><span style="border-top:2px dashed #ffb703; display:inline-block; width:34px; vertical-align:middle;"></span> likes</div>
      <div><span style="border-top:2px dashed #d62728; display:inline-block; width:34px; vertical-align:middle;"></span> bridge edge</div>
      <div><span style="border-top:2px solid #9e9e9e; display:inline-block; width:34px; vertical-align:middle;"></span> knows</div>
    """
    return add_html_legend(net.generate_html(notebook=False), legend_body)


def make_kmeans_network_html(edges_df: pd.DataFrame, results_df: pd.DataFrame) -> str:
    net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-40000, central_gravity=0.25, spring_length=150, spring_strength=0.02, damping=0.09)

    degree = get_degree_map(edges_df)
    # กัน error กรณีชื่อ node ซ้ำกันก่อนแปลงเป็น dict
    result_map = (
        results_df.drop_duplicates(subset=["node"], keep="first").set_index("node").to_dict("index")
        if not results_df.empty
        else {}
    )
    all_nodes = sorted(set(edges_df["source"]).union(set(edges_df["target"])))
    max_comm = int(results_df["communityId"].max()) if not results_df.empty else 0

    for node in all_nodes:
        info = result_map.get(node, {})
        community = int(info.get("communityId", -1)) if pd.notna(info.get("communityId", -1)) else -1
        node_type = str(info.get("node_type", "Person"))
        color = color_for_community(community, max_comm) if community >= 0 else "#999999"
        shape = "diamond" if node_type == "Food" else "dot"
        size = 16 + degree.get(node, 1) * 2
        title = (
            f"<b>{node}</b><br>"
            f"ประเภท: {node_type}<br>"
            f"K-means community: {community}<br>"
            f"Degree: {degree.get(node, 0)}<br>"
            f"Distance: {info.get('distanceFromCentroid', '')}<br>"
            f"Silhouette: {info.get('silhouette', '')}"
        )
        net.add_node(node, label=node, title=title, color=color, shape=shape, size=size)

    for row in edges_df.itertuples(index=False):
        if str(row.relation) == "likes":
            net.add_edge(row.source, row.target, title="likes", color="#ffb703", dashes=True, width=2, label="likes")
        else:
            net.add_edge(row.source, row.target, title=str(row.relation), color="#B0B0B0", label=str(row.relation))

    net.set_options('{"nodes":{"font":{"size":16}},"edges":{"smooth":false,"font":{"size":10,"align":"middle"}},"interaction":{"hover":true,"navigationButtons":true,"keyboard":true},"physics":{"enabled":true,"stabilization":{"iterations":250}}}')

    comm_items = []
    for cid in sorted(results_df["communityId"].dropna().astype(int).unique().tolist()):
        comm_items.append(f'<div><span style="display:inline-block; width:13px; height:13px; border-radius:50%; background:{color_for_community(cid, max_comm)}; margin-right:6px;"></span>Community {cid}</div>')
    legend_body = """
      <div style="height:12px; background:linear-gradient(90deg,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725); border-radius:6px; margin-bottom:8px;"></div>
      <div style="font-size:12px; color:#555; margin-bottom:8px;">สีไล่ตามหมายเลข community จาก K-means</div>
    """ + "".join(comm_items) + """
      <hr style="border:none; border-top:1px solid #eee; margin:8px 0;">
      <div><span style="font-size:16px;">●</span> Person / member</div>
      <div><span style="color:#ffb703; font-size:18px;">◆</span> Food node</div>
      <div><span style="border-top:2px dashed #ffb703; display:inline-block; width:34px; vertical-align:middle;"></span> likes</div>
    """
    return add_html_legend(net.generate_html(notebook=False), legend_body)



def make_imdb_network_html(edges_df: pd.DataFrame, results_df: pd.DataFrame) -> str:
    """วาด IMDb graph โดยใช้สีไล่ระดับตาม community และรูปร่างตามประเภท node เช่น Movie/Actor"""
    if edges_df.empty:
        # กรณีดึง edge จาก projection ไม่ได้ ให้สร้างกราฟจาก node อย่างเดียว
        all_nodes = results_df[["node", "node_type", "communityId"]].drop_duplicates()
        edges_df = pd.DataFrame(columns=["source", "target", "relation", "source_type", "target_type"])
    else:
        all_nodes = get_node_table(edges_df)

    # บาง dataset เช่น IMDb อาจมีชื่อ node ซ้ำกันได้
    # เช่น title/name ซ้ำกัน หรือคนกับหนังชื่อซ้ำกัน
    # ถ้า set_index("node") ทันที pandas จะ error: DataFrame index must be unique
    # จึงรวม/ตัดซ้ำให้เหลือ 1 แถวต่อชื่อ node ก่อนเอาไปทำ dict สำหรับ hover/สีของกราฟ
    if not results_df.empty:
        result_lookup_df = (
            results_df
            .sort_values(["communityId", "node"], kind="stable")
            .drop_duplicates(subset=["node"], keep="first")
        )
        result_map = result_lookup_df.set_index("node").to_dict("index")
    else:
        result_map = {}
    degree = get_degree_map(edges_df) if not edges_df.empty else {str(r.node): 0 for r in all_nodes.itertuples(index=False)}
    max_comm = int(results_df["communityId"].max()) if not results_df.empty else 0

    net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-50000, central_gravity=0.25, spring_length=170, spring_strength=0.02, damping=0.09)

    # รวม node จาก edge และจากผล K-means เผื่อ edge limit ทำให้ node บางตัวไม่ติดเข้ามา
    node_rows = []
    if not all_nodes.empty:
        for r in all_nodes.itertuples(index=False):
            node_rows.append({"node": str(r.node), "node_type": str(r.node_type)})
    for r in results_df[["node", "node_type"]].drop_duplicates().head(250).itertuples(index=False):
        node_rows.append({"node": str(r.node), "node_type": str(r.node_type)})
    node_df = pd.DataFrame(node_rows).drop_duplicates(subset=["node"]) if node_rows else pd.DataFrame(columns=["node", "node_type"])

    for row in node_df.itertuples(index=False):
        node = str(row.node)
        info = result_map.get(node, {})
        community = int(info.get("communityId", -1)) if pd.notna(info.get("communityId", -1)) else -1
        node_type = str(info.get("node_type", row.node_type))
        color = color_for_community(community, max_comm) if community >= 0 else "#999999"
        shape = "box" if "Movie" in node_type or "Film" in node_type else "dot"
        size = 14 + min(degree.get(node, 0), 12) * 2
        title = (
            f"<b>{node}</b><br>"
            f"ประเภท: {node_type}<br>"
            f"K-means community: {community}<br>"
            f"Degree ในรูป: {degree.get(node, 0)}<br>"
            f"Distance: {info.get('distanceFromCentroid', '')}<br>"
            f"Silhouette: {info.get('silhouette', '')}"
        )
        net.add_node(node, label=node, title=title, color=color, shape=shape, size=size)

    for row in edges_df.itertuples(index=False):
        net.add_edge(str(row.source), str(row.target), label=str(row.relation), title=str(row.relation), color="#B0B0B0", width=1)

    net.set_options('{"nodes":{"font":{"size":14}},"edges":{"smooth":false,"font":{"size":9,"align":"middle"}},"interaction":{"hover":true,"navigationButtons":true,"keyboard":true},"physics":{"enabled":true,"stabilization":{"iterations":350}}}')

    comm_items = []
    for cid in sorted(results_df["communityId"].dropna().astype(int).unique().tolist()):
        comm_items.append(f'<div><span style="display:inline-block; width:13px; height:13px; border-radius:50%; background:{color_for_community(cid, max_comm)}; margin-right:6px;"></span>Community {cid}</div>')
    legend_body = """
      <div style="height:12px; background:linear-gradient(90deg,#440154,#414487,#2a788e,#22a884,#7ad151,#fde725); border-radius:6px; margin-bottom:8px;"></div>
      <div style="font-size:12px; color:#555; margin-bottom:8px;">สีไล่ตาม community จาก IMDb K-means</div>
    """ + "".join(comm_items) + """
      <hr style="border:none; border-top:1px solid #eee; margin:8px 0;">
      <div><span style="font-size:16px;">●</span> Actor / person-like node</div>
      <div><span style="display:inline-block; width:14px; height:10px; background:#999; margin-right:4px;"></span> Movie / film-like node</div>
      <div><span style="border-top:2px solid #9e9e9e; display:inline-block; width:34px; vertical-align:middle;"></span> IMDb relationship</div>
    """
    return add_html_legend(net.generate_html(notebook=False), legend_body)


def plot_top10(df: pd.DataFrame, metric: str, title: str, name_col: str = "name") -> None:
    if df.empty or metric not in df.columns:
        st.warning(f"ไม่มีข้อมูล {metric}")
        return
    top = df.sort_values(metric, ascending=False).head(10).copy()
    fig = px.bar(
        top,
        x=metric,
        y=name_col,
        orientation="h",
        color=metric,
        color_continuous_scale="YlGnBu",
        title=title,
        hover_data=[c for c in ["node_type", "bridge_count"] if c in top.columns],
    )
    text_template = "%{x:.0f}" if metric == "bridge_count" else "%{x:.4f}"
    fig.update_traces(texttemplate=text_template, textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=520,
        margin=dict(l=10, r=60, t=60, b=10),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_explanation_box(title: str, bullets: List[str]) -> None:
    text = "\n".join([f"- {b}" for b in bullets])
    st.info(f"**{title}**\n\n{text}")


def show_kmeans_results(config: DatasetConfig, edges_df: pd.DataFrame, results_df: pd.DataFrame):
    st.subheader("ผลลัพธ์ K-means community detection")
    show_explanation_box(
        "อ่านกราฟนี้ยังไง",
        [
            "สีของ node คือ community ที่ Neo4j GDS K-means จัดกลุ่มให้",
            "node ที่อยู่สีเดียวกัน แปลว่ามีตำแหน่ง/รูปแบบความสัมพันธ์ใกล้กันใน embedding space",
            "Food node เป็นรูปเพชร ส่วน Person/member เป็นวงกลม",
            "ค่า silhouette ยิ่งสูง ยิ่งแปลว่า node นั้นเข้ากับ community ของตัวเองได้ดี",
        ],
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("จำนวน nodes", len(set(edges_df["source"]).union(set(edges_df["target"]))))
    col2.metric("จำนวน edges", len(edges_df))
    col3.metric("จำนวน communities", results_df["communityId"].nunique())

    summary = results_df.groupby("communityId", as_index=False).size().rename(columns={"size": "จำนวน node"})
    fig = px.bar(
        summary,
        x="communityId",
        y="จำนวน node",
        color="communityId",
        color_continuous_scale="Viridis",
        title="จำนวน node ในแต่ละ community",
    )
    fig.update_traces(text=summary["จำนวน node"], texttemplate="%{text}", textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=50, t=60, b=10), uniformtext_minsize=11, uniformtext_mode="show")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("กราฟแท่งนี้ใช้ดูว่าแต่ละ community มี node กี่ตัว ถ้ากลุ่มใดสูงมาก แปลว่ามีสมาชิกเยอะ")
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Interactive network graph")
    components.html(make_kmeans_network_html(edges_df, results_df), height=760, scrolling=True)

    st.markdown("### ตารางผลลัพธ์ราย node")
    st.caption("ตารางนี้ใช้ดูว่าแต่ละ node ถูกจัดเข้า community ไหน พร้อมระยะห่างจาก centroid และ silhouette")
    st.dataframe(results_df, use_container_width=True)
    st.download_button(
        "ดาวน์โหลดผลลัพธ์ K-means เป็น CSV",
        data=results_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{config.key}_kmeans_results.csv",
        mime="text/csv",
    )


def cypher_note(title: str, code: str, note: str) -> None:
    with st.expander(title, expanded=True):
        st.code(code.strip(), language="cypher")
        st.markdown(note)


# -----------------------------
# UI
# -----------------------------
st.title("🕸️ Neo4j GDS Network Visualizer")
st.caption("รวมงาน Network visualization + Centrality + K-means community detection โดยคำนวณด้วย Neo4j Graph Data Science")

with st.sidebar:
    st.header("ตั้งค่า Neo4j")
    uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    username = st.text_input("Username", value="neo4j")
    password = st.text_input("Password", value="12345678", type="password")
    database = st.text_input("Database", value=DEFAULT_DATABASE)

    st.header("ตั้งค่า Dataset")
    include_favorites = st.checkbox("รวม favorites_rows.csv เพื่อเพิ่มอาหารในกราฟ", value=True)
    st.caption("เปิดอันนี้แล้วกราฟจะมีทั้งคน + อาหาร และเส้น likes เหมือนตัวอย่างของอาจารย์")

    st.header("ตั้งค่า K-means")
    k_default = st.slider("จำนวนกลุ่ม k", min_value=2, max_value=8, value=3, step=1)
    embedding_dim = st.slider("Embedding dimension", min_value=4, max_value=32, value=8, step=4)
    random_seed = st.number_input("Random seed", min_value=1, value=42, step=1)
    imdb_edge_limit = st.slider("จำนวน IMDb edges ที่แสดงในกราฟ", min_value=100, max_value=1200, value=500, step=100)

    st.warning("บน Streamlit Cloud ห้ามใช้ localhost ต้องใช้ Neo4j Aura หรือ server ที่เข้าจากอินเทอร์เน็ตได้")

try:
    driver = get_driver(uri, username, password)
    gds_version = check_gds(driver, database)
    st.success(f"เชื่อมต่อ Neo4j สำเร็จ | GDS version: {gds_version}")
except Exception as exc:
    st.error("ยังเชื่อมต่อ Neo4j/GDS ไม่สำเร็จ ตรวจสอบ Docker, password และ GDS plugin ก่อน")
    st.code(str(exc))
    st.stop()

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    raw_upload = st.file_uploader("อัปโหลด edges_rows.csv หรือใช้ไฟล์ตัวอย่างในโฟลเดอร์", type=["csv"], key="edges_upload")
with col_upload2:
    fav_upload = st.file_uploader("อัปโหลด favorites_rows.csv หรือใช้ไฟล์ตัวอย่างในโฟลเดอร์", type=["csv"], key="favorites_upload")

try:
    raw_edges = read_default_or_upload(raw_upload, "edges_rows.csv")
    person_edges_df = normalize_edges_df(raw_edges)
except Exception as exc:
    st.error(f"อ่าน/เตรียม edges_rows.csv ไม่สำเร็จ: {exc}")
    st.stop()

try:
    raw_favorites = read_default_or_upload(fav_upload, "favorites_rows.csv")
    favorite_edges_df = normalize_favorites_df(raw_favorites)
except Exception as exc:
    st.warning(f"อ่าน favorites_rows.csv ไม่สำเร็จ จะแสดงเฉพาะ edges_rows.csv: {exc}")
    favorite_edges_df = pd.DataFrame(columns=["source", "target", "relation", "source_type", "target_type"])

graph_edges_df = build_graph_edges(person_edges_df, favorite_edges_df, include_favorites)
network_stats = get_basic_network_stats(graph_edges_df)

st.markdown("### ข้อมูลที่ใช้สร้างกราฟ")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodes", network_stats["nodes"])
c2.metric("Edges", network_stats["edges"])
c3.metric("Components", network_stats["components"])
c4.metric("Diameter", network_stats["diameter"])
st.caption("ถ้าเปิดรวม favorites_rows.csv แล้ว จำนวน edge จะเพิ่มจากความสัมพันธ์ likes ระหว่างคนกับอาหาร")

with st.expander("ดูตัวอย่าง edge list ที่ผ่านการ clean แล้ว", expanded=False):
    st.dataframe(graph_edges_df.head(30).copy(), use_container_width=True)
    st.download_button(
        "ดาวน์โหลด graph_edges_clean.csv",
        data=graph_edges_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="graph_edges_clean.csv",
        mime="text/csv",
    )

# Tabs
tab_visual, tab_cent, tab_k_edges, tab_k_karate, tab_k_imdb, tab_code = st.tabs([
    "1) Network Visualizer",
    "2) Centrality",
    "3) Ex.3 K-means: edges_rows",
    "4) Ex.4 K-means: karate_club",
    "5) Ex.5 K-means: IMDb",
    "6) Cypher + Notes",
])

with tab_visual:
    st.header("1) Network Visualizer")
    show_explanation_box(
        "Tab นี้แสดงอะไร",
        [
            "ใช้ดูภาพรวมว่าใครเชื่อมกับใคร และใครชอบอาหารอะไร",
            "edges_rows.csv สร้างเส้น knows ระหว่างคนกับคน",
            "favorites_rows.csv สร้าง node อาหาร และเส้น likes ระหว่างคนกับอาหาร",
            "สีของ node คนไล่จากอ่อนไปเข้มตาม Degree หรือจำนวน connection",
        ],
    )
    st.markdown("### Interactive network graph")
    components.html(make_visual_network_html(graph_edges_df), height=760, scrolling=True)

    st.markdown("### ตารางแยกประเภท node")
    node_summary = get_node_table(graph_edges_df).groupby("node_type", as_index=False).size().rename(columns={"size": "จำนวน node"})
    fig = px.bar(node_summary, x="node_type", y="จำนวน node", color="จำนวน node", color_continuous_scale="YlGnBu", title="จำนวน node แยกตามประเภท")
    fig.update_traces(text=node_summary["จำนวน node"], texttemplate="%{text}", textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=50, t=60, b=10), uniformtext_minsize=11, uniformtext_mode="show")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(node_summary, use_container_width=True)

with tab_cent:
    st.header("2) Centrality visualization")
    show_explanation_box(
        "Tab นี้แสดงอะไร",
        [
            "คำนวณความสำคัญของ node ด้วย Neo4j GDS ได้แก่ Betweenness, Bridges, Closeness, Degree, Eigenvector และ PageRank",
            "กราฟ interactive ใช้ขนาด node และสีเข้มเพื่อสื่อค่า Betweenness",
            "เส้นประสีแดงคือ bridge edge ถ้ามี หมายถึงเส้นที่ตัดออกแล้วทำให้เครือข่ายแยกส่วน",
            "Food node ถูกใส่เข้ามาจาก favorites_rows.csv เพื่อให้เห็นความสัมพันธ์ likes เหมือนตัวอย่างของอาจารย์",
        ],
    )

    if st.button("Run Centrality", type="primary"):
        try:
            with st.spinner("Import graph เข้า Neo4j..."):
                counts = import_edges_to_neo4j(driver, database, CENTRALITY_CONFIG, graph_edges_df)
            with st.spinner("สร้าง GDS graph projection..."):
                projection = project_graph(driver, database, CENTRALITY_CONFIG)
            with st.spinner("คำนวณ centrality ด้วย Neo4j GDS..."):
                centrality_df, bridges_df = calculate_centrality(driver, database, CENTRALITY_CONFIG)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nodes", counts["nodes"])
            c2.metric("Edges ใน DB", counts["edges"])
            c3.metric("Projection relationships", projection["relationshipCount"])
            c4.metric("Bridge edges", len(bridges_df))

            st.subheader("Interactive Network Graph")
            components.html(make_centrality_network_html(graph_edges_df, centrality_df, bridges_df), height=760, scrolling=True)

            st.subheader("ตารางผล Centrality จาก Neo4j GDS")
            st.caption("ตารางนี้เป็นค่าที่ Neo4j GDS คำนวณให้โดยตรง ไม่ได้คำนวณด้วย NetworkX")
            st.dataframe(centrality_df, use_container_width=True)
            st.download_button(
                "ดาวน์โหลด centrality_results.csv",
                data=centrality_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="neo4j_gds_centrality_results.csv",
                mime="text/csv",
            )
            st.download_button(
                "ดาวน์โหลด bridge_edges.csv",
                data=bridges_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="neo4j_gds_bridge_edges.csv",
                mime="text/csv",
            )

            tabs = st.tabs(["Betweenness", "Bridges", "Closeness", "Degree", "Eigenvector", "PageRank"])
            with tabs[0]:
                st.markdown("**Betweenness** = node ที่เป็นตัวกลาง/ทางผ่านของ shortest paths หลายเส้น ยิ่งสูงยิ่งเป็นสะพานเชื่อมกลุ่ม")
                plot_top10(centrality_df, "betweenness", "Top 10 Betweenness Centrality")
            with tabs[1]:
                st.markdown("**Bridge Count** = จำนวน bridge edge ที่เกี่ยวกับ node นั้น ถ้าเป็น 0 แปลว่าไม่มีเส้นสำคัญที่ตัดแล้วกราฟแตก")
                if centrality_df["bridge_count"].sum() == 0:
                    st.warning("ไม่พบ bridge edge ในกราฟนี้ ดังนั้น bridge count ของทุก node เป็น 0")
                plot_top10(centrality_df, "bridge_count", "Top 10 Bridge Count")
            with tabs[2]:
                st.markdown("**Closeness** = node ที่เข้าถึง node อื่นได้เร็ว ใช้จำนวน step เฉลี่ยน้อย")
                plot_top10(centrality_df, "closeness", "Top 10 Closeness Centrality")
            with tabs[3]:
                st.markdown("**Degree** = node ที่มีจำนวน connection มาก ยิ่งสูงยิ่งเชื่อมกับหลาย node")
                plot_top10(centrality_df, "degree", "Top 10 Degree Centrality")
            with tabs[4]:
                st.markdown("**Eigenvector** = node ที่เชื่อมกับ node สำคัญอื่น ๆ ยิ่งสูงยิ่งอยู่ในกลุ่มที่มีอิทธิพล")
                plot_top10(centrality_df, "eigenvector", "Top 10 Eigenvector Centrality")
            with tabs[5]:
                st.markdown("**PageRank** = วัดความสำคัญจากคุณภาพของ connection คล้ายแนวคิดหน้าเว็บที่ถูกอ้างถึงโดยหน้าเว็บสำคัญ")
                plot_top10(centrality_df, "pagerank", "Top 10 PageRank")
        except Exception as exc:
            st.error("เกิด error ตอนรัน Centrality")
            st.exception(exc)

with tab_k_edges:
    st.header("3) Ex.3 Apply K-means for community detection on edges_rows.csv")
    show_explanation_box(
        "Tab นี้แสดงอะไร",
        [
            "ใช้ Neo4j GDS ทำ community detection แบบ K-means",
            "ก่อนทำ K-means จะใช้ FastRP แปลงตำแหน่งของ node ในกราฟให้เป็น embedding vector",
            "จากนั้นใช้ gds.kmeans.stream แบ่ง node เป็น k กลุ่ม",
            "ถ้าเปิดรวม favorites_rows.csv จะมี food node เข้ามาช่วยให้เห็น community ตามอาหารที่ชอบด้วย",
        ],
    )
    st.markdown("Workflow: graph projection → FastRP embedding → GDS K-means")

    if st.button("Run K-means on edges_rows graph", type="primary"):
        try:
            with st.spinner("Import graph เข้า Neo4j และคำนวณ K-means ด้วย GDS..."):
                results_df, import_counts, projection_info, embedding_info = run_kmeans_pipeline(
                    driver, database, KMEANS_EDGES_CONFIG, graph_edges_df, int(k_default), int(embedding_dim), int(random_seed)
                )
            with st.expander("รายละเอียดขั้นตอนที่ Neo4j คืนค่า", expanded=False):
                st.write("Import counts:", import_counts)
                st.write("Projection:", projection_info)
                st.write("FastRP:", embedding_info)
            show_kmeans_results(KMEANS_EDGES_CONFIG, graph_edges_df, results_df)
        except Exception as exc:
            st.error("เกิด error ตอนรัน Ex.3")
            st.exception(exc)

with tab_k_karate:
    st.header("4) Ex.4 Apply K-means for community detection on karate_club")
    show_explanation_box(
        "Tab นี้แสดงอะไร",
        [
            "ใช้กราฟตัวอย่าง Zachary's Karate Club ซึ่งเป็น network ของสมาชิกชมรมคาราเต้",
            "Neo4j GDS คำนวณ FastRP embedding แล้วใช้ K-means แบ่ง community",
            "กราฟสีเดียวกันหมายถึงสมาชิกถูกจัดอยู่ community เดียวกัน",
            "ตาราง cross-tab ใช้เทียบ community ที่ K-means หาได้กับ original club label",
        ],
    )
    karate_edges_df, karate_nodes_df = make_karate_edges()

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Karate club edge list")
        st.dataframe(karate_edges_df.head(20).copy(), use_container_width=True)
    with col_b:
        st.write("Original club label")
        st.dataframe(karate_nodes_df.head(20).copy(), use_container_width=True)

    st.write(f"Karate club มี {karate_nodes_df.shape[0]} nodes และ {karate_edges_df.shape[0]} edges")

    if st.button("Run K-means on karate_club", type="primary"):
        try:
            with st.spinner("สร้าง karate_club ใน Neo4j และคำนวณ K-means ด้วย GDS..."):
                results_df, import_counts, projection_info, embedding_info = run_kmeans_pipeline(
                    driver, database, KARATE_CONFIG, karate_edges_df, int(k_default), int(embedding_dim), int(random_seed)
                )
                add_karate_club_property(driver, database, karate_nodes_df)
                results_with_club = results_df.merge(karate_nodes_df, on="node", how="left")

            with st.expander("รายละเอียดขั้นตอนที่ Neo4j คืนค่า", expanded=False):
                st.write("Import counts:", import_counts)
                st.write("Projection:", projection_info)
                st.write("FastRP:", embedding_info)
            show_kmeans_results(KARATE_CONFIG, karate_edges_df, results_with_club)

            st.markdown("### เปรียบเทียบ K-means community กับ original club")
            st.caption("ถ้า community ที่คำนวณได้สอดคล้องกับ club เดิม จะเห็นค่ากระจุกตัวในตารางนี้")
            compare = pd.crosstab(results_with_club["communityId"], results_with_club["club"])
            st.dataframe(compare, use_container_width=True)
        except Exception as exc:
            st.error("เกิด error ตอนรัน Ex.4")
            st.exception(exc)


with tab_k_imdb:
    st.header("5) Ex.5 Apply K-means for community detection on imdb")
    show_explanation_box(
        "Tab นี้แสดงอะไร",
        [
            "ทำตามคำสั่งอาจารย์โดยใช้ GDS Python Client: G = gds.graph.load_imdb()",
            "โหลด sample graph IMDb เข้ามาเป็น graph projection ของ Neo4j GDS",
            "ใช้ FastRP สร้าง embedding ของ node แล้วใช้ gds.kmeans.stream แบ่ง community",
            "กราฟใช้สีไล่ระดับตาม community และมีตัวเลขบนกราฟแท่งเพื่อให้อ่านง่ายขึ้น",
        ],
    )
    st.markdown("Workflow: `gds.graph.load_imdb()` → FastRP embedding → GDS K-means → Visualization")
    st.warning("Tab นี้ต้องมี package `graphdatascience` และ Neo4j GDS ใช้งานได้แล้ว")

    if st.button("Run K-means on IMDb sample graph", type="primary"):
        try:
            with st.spinner("โหลด IMDb ด้วย gds.graph.load_imdb() และคำนวณ K-means ด้วย Neo4j GDS..."):
                imdb_results_df, imdb_edges_df, imdb_graph_info, imdb_embedding_info = run_imdb_pipeline(
                    driver,
                    database,
                    uri,
                    username,
                    password,
                    int(k_default),
                    int(embedding_dim),
                    int(random_seed),
                    int(imdb_edge_limit),
                )

            st.success("โหลด IMDb และคำนวณ K-means สำเร็จ")
            with st.expander("รายละเอียดจาก Neo4j/GDS", expanded=False):
                st.write("Graph info:", imdb_graph_info)
                st.write("FastRP:", imdb_embedding_info)

            c1, c2, c3 = st.columns(3)
            c1.metric("Nodes ที่ได้ผล K-means", len(imdb_results_df))
            c2.metric("Edges ที่นำมาแสดง", len(imdb_edges_df))
            c3.metric("จำนวน communities", imdb_results_df["communityId"].nunique())

            st.markdown("### จำนวน node ในแต่ละ IMDb community")
            summary = imdb_results_df.groupby("communityId", as_index=False).size().rename(columns={"size": "จำนวน node"})
            fig = px.bar(
                summary,
                x="communityId",
                y="จำนวน node",
                color="communityId",
                color_continuous_scale="Viridis",
                title="IMDb: จำนวน node ในแต่ละ community",
            )
            fig.update_traces(text=summary["จำนวน node"], texttemplate="%{text}", textposition="outside", cliponaxis=False)
            fig.update_layout(margin=dict(l=10, r=50, t=60, b=10), uniformtext_minsize=11, uniformtext_mode="show")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Interactive IMDb graph")
            st.caption("เพื่อไม่ให้กราฟหนักเกินไป ระบบแสดง edge ตามจำนวนที่ตั้งไว้ใน sidebar แต่ K-means คำนวณจาก IMDb graph projection")
            components.html(make_imdb_network_html(imdb_edges_df, imdb_results_df), height=760, scrolling=True)

            st.markdown("### ตารางผลลัพธ์ IMDb K-means")
            st.dataframe(imdb_results_df, use_container_width=True)
            st.download_button(
                "ดาวน์โหลด imdb_kmeans_results.csv",
                data=imdb_results_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="imdb_kmeans_results.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error("เกิด error ตอนรัน IMDb K-means")
            st.exception(exc)

with tab_code:
    st.header("6) Cypher หลักที่ใช้ + Notes")
    st.markdown("Tab นี้อธิบายว่า query หลักแต่ละตัวใช้เพื่อสื่ออะไรในงาน")

    cypher_note(
        "A) สร้าง Graph Projection",
        """
CALL gds.graph.project(
  'graphName',
  'NodeLabel',
  {REL_TYPE: {orientation: 'UNDIRECTED'}}
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;
        """,
        """
**สื่ออะไร:** แปลง node/relationship ที่อยู่ใน Neo4j database ให้เป็น graph ใน memory ของ GDS ก่อน เพราะ algorithm ของ GDS ต้องรันบน graph projection ไม่ใช่รันตรงจากตาราง database ธรรมดา  
**ทำไม UNDIRECTED:** ความสัมพันธ์ใน social network เช่น knows/likes ในงานนี้ตีความเป็นความเชื่อมโยงสองทางเพื่อวิเคราะห์โครงสร้างเครือข่าย
        """,
    )

    cypher_note(
        "B) Betweenness Centrality",
        """
CALL gds.betweenness.stream('graphName')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS node, score
ORDER BY score DESC;
        """,
        """
**สื่ออะไร:** หา node ที่ทำหน้าที่เป็นตัวกลางของเส้นทางสั้นที่สุดในเครือข่าย  
**แปลผลง่าย ๆ:** score สูง = คน/อาหาร node นั้นเป็นจุดผ่านสำคัญ ถ้าขาดไป network อาจเชื่อมกันยากขึ้น
        """,
    )

    cypher_note(
        "C) Bridges",
        """
CALL gds.bridges.stream('graphName')
YIELD from, to
RETURN gds.util.asNode(from).name AS source,
       gds.util.asNode(to).name AS target;
        """,
        """
**สื่ออะไร:** หา edge ที่เป็นสะพานเชื่อมส่วนต่าง ๆ ของกราฟ  
**แปลผลง่าย ๆ:** ถ้า edge นี้ถูกลบแล้วกราฟแตกเป็นหลาย component มากขึ้น แปลว่าเป็น bridge edge
        """,
    )

    cypher_note(
        "D) Centrality ตัวอื่น ๆ",
        """
CALL gds.closeness.stream('graphName') YIELD nodeId, score;
CALL gds.degree.stream('graphName') YIELD nodeId, score;
CALL gds.eigenvector.stream('graphName') YIELD nodeId, score;
CALL gds.pageRank.stream('graphName') YIELD nodeId, score;
        """,
        """
**Closeness:** ใครเข้าถึงคนอื่นได้เร็ว  
**Degree:** ใครมี connection เยอะ  
**Eigenvector:** ใครเชื่อมกับ node ที่สำคัญ  
**PageRank:** ใครมีความสำคัญจากคุณภาพของ connection ที่เชื่อมเข้ามา
        """,
    )

    cypher_note(
        "E) FastRP Embedding ก่อนทำ K-means",
        """
CALL gds.fastRP.mutate('graphName', {
  embeddingDimension: 8,
  randomSeed: 42,
  mutateProperty: 'embedding'
})
YIELD nodePropertiesWritten;
        """,
        """
**สื่ออะไร:** แปลง node ใน network ให้กลายเป็นเวกเตอร์ตัวเลข หรือ embedding  
**ทำไมต้องทำ:** K-means ต้องการข้อมูลตัวเลขเป็น input ดังนั้นต้องสร้าง embedding จากโครงสร้างกราฟก่อน
        """,
    )

    cypher_note(
        "F) K-means Community Detection",
        """
CALL gds.kmeans.stream('graphName', {
  nodeProperty: 'embedding',
  k: 3,
  randomSeed: 42,
  initialSampler: 'kmeans++',
  computeSilhouette: true
})
YIELD nodeId, communityId, distanceFromCentroid, silhouette
RETURN gds.util.asNode(nodeId).name AS node,
       communityId,
       distanceFromCentroid,
       silhouette;
        """,
        """
**สื่ออะไร:** แบ่ง node เป็น community ตามความใกล้กันของ embedding  
**แปลผลง่าย ๆ:** node ที่อยู่ community เดียวกัน มีโครงสร้างความสัมพันธ์ใกล้เคียงกันใน network
        """,
    )


    cypher_note(
        "G) IMDb sample graph ด้วย GDS Python Client",
        """
# ใน Python / Streamlit ใช้คำสั่งนี้ตามโจทย์อาจารย์
from graphdatascience import GraphDataScience

gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"), database="neo4j")
G = gds.graph.load_imdb()
        """,
        """
**สื่ออะไร:** โหลด sample graph IMDb ที่ Neo4j GDS Python Client เตรียมไว้ให้ โดยเก็บเป็น graph projection สำหรับรัน algorithm ต่อ  
**เอาไปใช้ต่อยังไง:** หลังได้ `G` แล้ว โปรแกรมสร้าง FastRP embedding และส่งเข้า K-means เหมือนกับกราฟ edges_rows และ karate_club  
**เหตุผลที่ใช้ IMDb:** เป็นตัวอย่าง graph คน/หนังที่มีความสัมพันธ์ชัดเจน เหมาะกับการทดลอง community detection ว่า node กลุ่มไหนมีโครงสร้างใกล้กัน
        """,
    )
