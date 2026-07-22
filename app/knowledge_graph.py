"""
Knowledge graph construction and persistence.

Nodes: entities (equipment, documents, personnel, regulatory refs, dates)
       + document nodes (the source files themselves)
Edges: co-occurrence within a document (entity <-> entity) and
       containment (document -> entity), with weights = co-occurrence count.

Persisted as JSON so the whole graph survives process restarts without
needing an external graph DB for the prototype.
"""
import json
import os
import threading
from itertools import combinations
from pathlib import Path

import networkx as nx

from app.entity_extraction import Entity, extract_entities, extract_document_metadata

GRAPH_PATH = Path(__file__).parent.parent / "data" / "graph_store.json"
DOCS_PATH = Path(__file__).parent.parent / "data" / "documents_store.json"

_lock = threading.Lock()


class KnowledgeGraphStore:
    def __init__(self):
        self.graph = nx.Graph()
        self.documents = {}  # doc_id -> {filename, upload_date, metadata, chunks, entities}
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if GRAPH_PATH.exists():
            data = json.loads(GRAPH_PATH.read_text())
            self.graph = nx.node_link_graph(data, edges="edges")
        if DOCS_PATH.exists():
            self.documents = json.loads(DOCS_PATH.read_text())

    def _save(self):
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRAPH_PATH.write_text(json.dumps(nx.node_link_data(self.graph, edges="edges")))
        DOCS_PATH.write_text(json.dumps(self.documents, indent=2))

    # ---------- ingestion ----------
    def ingest_document(self, doc_id: str, filename: str, text: str, chunks: list, upload_date: str):
        with _lock:
            entities = extract_entities(text)
            metadata = extract_document_metadata(text)

            doc_node = f"DOCUMENT_FILE:{doc_id}"
            metadata = {k: v for k, v in metadata.items() if k != "document_id"}
            self.graph.add_node(
                doc_node,
                node_type="document",
                label=filename,
                document_id=doc_id,
                **metadata,
            )

            entity_keys = []
            for ent in entities:
                key = ent.key()
                entity_keys.append(key)
                if self.graph.has_node(key):
                    self.graph.nodes[key]["mentions"] = self.graph.nodes[key].get("mentions", 0) + 1
                    docs = set(self.graph.nodes[key].get("documents", []))
                    docs.add(doc_id)
                    self.graph.nodes[key]["documents"] = list(docs)
                else:
                    self.graph.add_node(
                        key,
                        node_type="entity",
                        label=ent.text,
                        entity_type=ent.label,
                        mentions=1,
                        documents=[doc_id],
                    )
                # containment edge: document -> entity
                if self.graph.has_edge(doc_node, key):
                    self.graph[doc_node][key]["weight"] += 1
                else:
                    self.graph.add_edge(doc_node, key, weight=1, edge_type="contains")

            # co-occurrence edges between entities appearing in the same document
            for a, b in combinations(sorted(set(entity_keys)), 2):
                if self.graph.has_edge(a, b):
                    self.graph[a][b]["weight"] += 1
                else:
                    self.graph.add_edge(a, b, weight=1, edge_type="co_occurs")

            self.documents[doc_id] = {
                "filename": filename,
                "upload_date": upload_date,
                "metadata": metadata,
                "chunks": chunks,
                "entity_count": len(entities),
                "entities": [{"text": e.text, "label": e.label} for e in entities],
                "full_text": text,
            }
            self._save()
            return {
                "doc_id": doc_id,
                "entities_extracted": len(entities),
                "unique_entities": len(set(entity_keys)),
                "metadata": metadata,
            }

    # ---------- query ----------
    def get_graph_json(self, entity_type_filter=None, search=None):
        nodes = []
        for n, d in self.graph.nodes(data=True):
            if entity_type_filter and d.get("entity_type") != entity_type_filter and d.get("node_type") != "document":
                continue
            if search and search.lower() not in d.get("label", "").lower():
                continue
            nodes.append({"id": n, **d})
        node_ids = {n["id"] for n in nodes}
        edges = []
        for u, v, d in self.graph.edges(data=True):
            if u in node_ids and v in node_ids:
                edges.append({"source": u, "target": v, **d})
        return {"nodes": nodes, "edges": edges}

    def get_entity_detail(self, key: str):
        if not self.graph.has_node(key):
            return None
        node = dict(self.graph.nodes[key])
        neighbors = []
        for nb in self.graph.neighbors(key):
            nb_data = dict(self.graph.nodes[nb])
            edge_data = dict(self.graph[key][nb])
            neighbors.append({"id": nb, **nb_data, "edge": edge_data})
        neighbors.sort(key=lambda x: x["edge"].get("weight", 0), reverse=True)
        return {"id": key, **node, "neighbors": neighbors}

    def search_documents(self, query: str, top_k: int = 5):
        """Lightweight TF-IDF style keyword search over chunks with citation."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_chunks = []
        chunk_meta = []
        for doc_id, doc in self.documents.items():
            for i, c in enumerate(doc["chunks"]):
                all_chunks.append(c["text"])
                chunk_meta.append({"doc_id": doc_id, "filename": doc["filename"], "chunk_index": i, "text": c["text"]})

        if not all_chunks:
            return []

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(all_chunks + [query])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            if sims[idx] <= 0:
                continue
            meta = chunk_meta[idx]
            results.append({
                "score": float(sims[idx]),
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "chunk_index": meta["chunk_index"],
                "snippet": meta["text"][:400],
            })
        return results

    def stats(self):
        entity_type_counts = {}
        for n, d in self.graph.nodes(data=True):
            if d.get("node_type") == "entity":
                et = d.get("entity_type", "OTHER")
                entity_type_counts[et] = entity_type_counts.get(et, 0) + 1
        return {
            "documents": len(self.documents),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_type_counts": entity_type_counts,
        }

    def reset(self):
        with _lock:
            self.graph = nx.Graph()
            self.documents = {}
            self._save()


store = KnowledgeGraphStore()
