
def dfs(environment, start, goal):

    # 1. Initialize the stack
    stack = [start]

    # 2. Track visited nodes
    visited = {start}

    # 3. Store parent information
    parent = {start: None}

    # 4. Store exploration order
    visited_order = []

    # 5. Store animation steps
    steps = []

    # 6. Continue while stack is not empty
    while stack:

        # Remove the top element
        current = stack.pop()

        # Record the current node
        visited_order.append(current)

        # Check whether goal is reached
        if current == goal:

            steps.append({
                "current": current,
                "explored": visited_order.copy(),
                "frontier": stack.copy()
            })

            break

        # Get neighbouring nodes
        neighbors = environment.get_neighbors(current)

        # Handle Graph and Maze environments
        if isinstance(neighbors, dict):
            neighbors = list(neighbors.keys())

        # Reverse neighbours to preserve left-to-right order
        for neighbor in reversed(neighbors):

            if neighbor not in visited:

                visited.add(neighbor)

                parent[neighbor] = current

                stack.append(neighbor)

        # Record current search state
        steps.append({
            "current": current,
            "explored": visited_order.copy(),
            "frontier": stack.copy()
        })

    # 7. Reconstruct the path
    path = []

    if goal in visited_order:

        current = goal

        while current is not None:

            path.append(current)

            current = parent[current]

        path.reverse()

    # 8. Return search results
    return {
        "found": goal in visited_order,
        "path": path,
        "visited_order": visited_order,
        "explored_count": len(visited_order),
        "steps": steps
    }


if __name__ == "__main__":

    from environments.graph import Graph

    graph = Graph()

    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("B", "E")
    graph.add_edge("C", "F")
    graph.add_edge("C", "G")

    result = dfs(graph, "A", "G")

    print("\nDFS Exploration Order:")
    print(result["visited_order"])

    print("\nFinal Path:")
    print(result["path"])

    print("\nGoal Found:")
    print(result["found"])

    print("\nExplored Nodes:")
    print(result["explored_count"])


    from environments.maze import Maze

    # Create maze
    grid = [
        ["S", ".", ".", "#", "."],
        ["#", "#", ".", "#", "."],
        [".", ".", ".", ".", "."],
        [".", "#", "#", "#", "."],
        [".", ".", ".", "G", "."]
    ]

    maze = Maze(grid)

    # Execute DFS
    result = dfs(maze, maze.start, maze.goal)

    print("\n========== DFS MAZE TEST ==========")

    print("\nStart Position:")
    print(maze.start)

    print("\nGoal Position:")
    print(maze.goal)

    print("\nExploration Order:")
    print(result["visited_order"])

    print("\nFinal Path:")
    print(result["path"])

    print("\nGoal Found:")
    print(result["found"])

    print("\nExplored Cells:")
    print(result["explored_count"])

