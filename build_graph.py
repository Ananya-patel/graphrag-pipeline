import networkx as nx
import pickle
from pathlib import Path
from pyvis.network import Network
import json


# ---- Build graph from triples ----

def build_graph(triples):
    """
    Convert triples into a NetworkX directed graph.
    
    Directed = edges have direction.
    Japan --influenced_by--> China
    is different from
    China --influenced_by--> Japan
    """
    G = nx.DiGraph()  # Directed Graph

    for triple in triples:
        subject = triple["subject"]
        relation = triple["relation"]
        obj = triple["object"]

        # Add nodes if they don't exist
        G.add_node(subject)
        G.add_node(obj)

        # Add edge with relation as attribute
        # If edge already exists, append relation
        if G.has_edge(subject, obj):
            existing = G[subject][obj]["relation"]
            if relation not in existing:
                G[subject][obj]["relation"] = (
                    existing + f", {relation}"
                )
        else:
            G.add_edge(subject, obj, relation=relation)

    return G


def save_graph(G, path="knowledge_graph.pkl"):
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")


def load_graph(path="knowledge_graph.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- Graph queries ----

def get_neighbors(G, entity, depth=1):
    """
    Get all entities connected to a given entity.
    depth=1: direct connections only
    depth=2: connections of connections (2-hop)
    """
    entity_title = entity.title()

    if entity_title not in G:
        # Try partial match
        matches = [n for n in G.nodes()
                   if entity.lower() in n.lower()]
        if not matches:
            return [], []
        entity_title = matches[0]
        print(f"  Matched '{entity}' to '{entity_title}'")

    if depth == 1:
        # Direct neighbors only
        outgoing = [
            (entity_title, G[entity_title][n]["relation"], n)
            for n in G.successors(entity_title)
        ]
        incoming = [
            (n, G[n][entity_title]["relation"], entity_title)
            for n in G.predecessors(entity_title)
        ]
        return outgoing, incoming

    elif depth == 2:
        # 2-hop: neighbors of neighbors
        all_triples = []
        visited = {entity_title}

        # First hop
        first_hop = list(G.successors(entity_title))
        for neighbor in first_hop:
            rel = G[entity_title][neighbor]["relation"]
            all_triples.append((entity_title, rel, neighbor))
            visited.add(neighbor)

            # Second hop
            for second in G.successors(neighbor):
                if second not in visited:
                    rel2 = G[neighbor][second]["relation"]
                    all_triples.append((neighbor, rel2, second))
                    visited.add(second)

        return all_triples, []


def find_path(G, start, end):
    """
    Find how two entities connect through the graph.
    This is the multi-hop question answering capability.
    """
    start_t = start.title()
    end_t = end.title()

    # Fuzzy match
    if start_t not in G:
        matches = [n for n in G.nodes()
                   if start.lower() in n.lower()]
        if matches:
            start_t = matches[0]

    if end_t not in G:
        matches = [n for n in G.nodes()
                   if end.lower() in n.lower()]
        if matches:
            end_t = matches[0]

    try:
        path = nx.shortest_path(G, start_t, end_t)
        # Build path with relations
        path_with_relations = []
        for i in range(len(path) - 1):
            rel = G[path[i]][path[i+1]]["relation"]
            path_with_relations.append(
                (path[i], rel, path[i+1])
            )
        return path_with_relations
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []


def get_most_connected(G, top_n=10):
    """Find the most connected entities in the graph."""
    degree_dict = dict(G.degree())
    sorted_nodes = sorted(
        degree_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_nodes[:top_n]


# ---- Visualize graph ----

def visualize_graph(G, output_path="graph.html", max_nodes=50):
    """
    Create an interactive HTML visualization of the graph.
    Opens in browser — nodes are draggable.
    """
    # Limit to most connected nodes for clarity
    if G.number_of_nodes() > max_nodes:
        degree_dict = dict(G.degree())
        top_nodes = sorted(
            degree_dict,
            key=degree_dict.get,
            reverse=True
        )[:max_nodes]
        G_viz = G.subgraph(top_nodes)
    else:
        G_viz = G

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0a0a0f",
        font_color="white",
        directed=True
    )

    net.from_nx(G_viz)

    # Style nodes by connection count
    for node in net.nodes:
        degree = G.degree(node["id"])
        # More connections = bigger node
        node["size"] = 10 + (degree * 3)
        node["title"] = f"{node['id']} ({degree} connections)"

        # Color by degree
        if degree >= 8:
            node["color"] = "#6ee7b7"   # green = highly connected
        elif degree >= 4:
            node["color"] = "#818cf8"   # purple = medium
        else:
            node["color"] = "#64748b"   # gray = low

    net.save_graph(output_path)
    print(f"Graph visualization saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Load triples
    print("Loading triples...")
    with open("triples.pkl", "rb") as f:
        triples = pickle.load(f)

    # Build graph
    print("Building graph...")
    G = build_graph(triples)
    save_graph(G)

    # Stats
    print(f"\n=== GRAPH STATS ===")
    print(f"Nodes (entities):  {G.number_of_nodes()}")
    print(f"Edges (relations): {G.number_of_edges()}")

    # Most connected entities
    print(f"\n=== MOST CONNECTED ENTITIES ===")
    for node, degree in get_most_connected(G, top_n=10):
        print(f"  {node}: {degree} connections")

    # Test neighbor lookup
    print(f"\n=== NEIGHBORS OF 'Japan' ===")
    outgoing, incoming = get_neighbors(G, "Japan", depth=1)
    print("Japan connects TO:")
    for s, r, o in outgoing[:8]:
        print(f"  → {o} [{r}]")
    print("Things that connect TO Japan:")
    for s, r, o in incoming[:8]:
        print(f"  ← {s} [{r}]")

    # Test path finding
    print(f"\n=== PATH: Buddhism → Japan ===")
    path = find_path(G, "Buddhism", "Japan")
    if path:
        for s, r, o in path:
            print(f"  {s} --[{r}]--> {o}")
    else:
        print("  No path found")

    print(f"\n=== PATH: China → Japanese Writing ===")
    path = find_path(G, "China", "Japanese Writing")
    if path:
        for s, r, o in path:
            print(f"  {s} --[{r}]--> {o}")
    else:
        print("  No direct path — trying kanji...")
        path = find_path(G, "China", "Kanji")
        for s, r, o in path:
            print(f"  {s} --[{r}]--> {o}")

    # Visualize
    print(f"\nCreating interactive visualization...")
    visualize_graph(G, "graph.html")
    print("Open graph.html in your browser to explore!")