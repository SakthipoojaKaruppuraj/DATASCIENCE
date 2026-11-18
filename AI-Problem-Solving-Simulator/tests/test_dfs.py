
from algorithms.search.dfs import dfs
from environments.graph import Graph
from environments.maze import Maze


# TEST 1: Start and Goal are the Same

def test_start_equals_goal():

    graph = Graph()
    graph.add_node("A")

    result = dfs(graph, "A", "A")

    assert result["found"] == True
    assert result["path"] == ["A"]
    assert result["explored_count"] == 1


# TEST 2: No Path Exists in Maze

def test_no_path_in_maze():

    grid = [
        ["S", "#", "G"]
    ]

    maze = Maze(grid)

    result = dfs(maze, maze.start, maze.goal)

    assert result["found"] == False
    assert result["path"] == []


# TEST 3: Unreachable Goal in Graph

def test_unreachable_goal():

    graph = Graph()

    graph.add_edge("A", "B")
    graph.add_edge("C", "D")

    result = dfs(graph, "A", "D")

    assert result["found"] == False
    assert result["path"] == []
