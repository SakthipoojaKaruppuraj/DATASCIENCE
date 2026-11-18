
from collections import deque


def bfs(environment, start, goal):

    # 1. Initialize queue
    queue = deque([start])

    # 2. Track discovered nodes
    visited = {start}

    # 3. Track parent nodes
    parent = {start: None}

    # 4. Track exploration order
    visited_order = []

    # 5. Store every animation step
    steps = []

    while queue:

        # Remove the first node
        current = queue.popleft()

        # Record explored node
        visited_order.append(current)

        # Get neighbours
        neighbors = environment.get_neighbors(current)

        if isinstance(neighbors, dict):
            neighbors = neighbors.keys()

        # Add unvisited neighbours
        if current != goal:

            for neighbor in neighbors:

                if neighbor not in visited:

                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        # Save current search state
        steps.append({
            "current": current,
            "explored": visited_order.copy(),
            "frontier": list(queue)
        })

        # Stop when goal is reached
        if current == goal:
            break

    # 6. Reconstruct final path
    path = []

    if goal in visited:

        current = goal

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()

    # 7. Return all results
    return {
        "found": goal in visited and goal in visited_order,
        "path": path,
        "visited_order": visited_order,
        "explored_count": len(visited_order),
        "steps": steps
    }



if __name__ == "__main__":

    from environments.graph import Graph

    print("\n========== BFS GRAPH TEST ==========")

    graph = Graph()

    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "E")
    graph.add_edge("D", "F")
    graph.add_edge("E", "F")

    result = bfs(graph, "A", "F")

    print("Start Node: A")
    print("Goal Node: F")

    print("\nExploration Order:")
    print(result["visited_order"])

    print("\nFinal Path:")
    print(result["path"])

    print("\nGoal Found:")
    print(result["found"])

    print("\nExplored Nodes:")
    print(result["explored_count"])


    from environments.maze import Maze

    print("\n========== BFS MAZE TEST ==========")

    grid = [
        ["S", ".", ".", "#", "."],
        ["#", "#", ".", "#", "."],
        [".", ".", ".", ".", "."],
        [".", "#", "#", "#", "."],
        [".", ".", ".", "G", "."]
    ]

    maze = Maze(grid)

    result = bfs(
        maze,
        maze.start,
        maze.goal
    )

    print("Start Position:", maze.start)
    print("Goal Position:", maze.goal)

    print("\nExploration Order:")
    print(result["visited_order"])

    print("\nFinal Path:")
    print(result["path"])

    print("\nGoal Found:")
    print(result["found"])

    print("\nExplored Cells:")
    print(result["explored_count"])

