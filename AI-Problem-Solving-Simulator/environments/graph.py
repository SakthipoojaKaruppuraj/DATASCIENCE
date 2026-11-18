
class Graph:

    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, node1, node2, weight=1):
        self.add_node(node1)
        self.add_node(node2)

        self.graph[node1][node2] = weight

        if not self.directed:
            self.graph[node2][node1] = weight

    def get_neighbors(self, node):
        return self.graph.get(node, {})

    def get_weight(self, node1, node2):
        return self.graph[node1][node2]

    def display(self):
        for node, neighbors in self.graph.items():
            print(f"{node} -> {neighbors}")


# Test the Graph class
if __name__ == "__main__":

    graph = Graph()

    graph.add_edge("A", "B", 4)
    graph.add_edge("A", "C", 2)
    graph.add_edge("B", "D", 5)
    graph.add_edge("C", "D", 3)

    print("\nGraph Representation:")
    graph.display()

    print("\nNeighbours of A:")
    print(graph.get_neighbors("A"))

    print("\nWeight between A and B:")
    print(graph.get_weight("A", "B"))

    print("\n--- DIRECTED GRAPH TEST ---")

directed_graph = Graph(directed=True)

directed_graph.add_edge("A", "B", 10)
directed_graph.add_edge("B", "C", 20)

directed_graph.display()

print("A to B:", directed_graph.get_neighbors("A"))
print("B to A:", directed_graph.get_neighbors("B"))


print("\n--- WEIGHTED GRAPH TEST ---")

weighted_graph = Graph()

weighted_graph.add_edge("A", "B", 10)
weighted_graph.add_edge("A", "C", 5)
weighted_graph.add_edge("B", "D", 15)

weighted_graph.display()
