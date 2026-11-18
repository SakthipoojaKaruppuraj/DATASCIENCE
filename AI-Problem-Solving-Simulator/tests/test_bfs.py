
from algorithms.search.bfs import bfs
from environments.graph import Graph
from environments.maze import Maze


# TEST 1: Start and Goal are the Same

def test_start_equals_goal():

    graph = Graph()

    graph.add_node("A")

    result = bfs(graph, "A", "A")

    assert result["found"] == True
    assert result["path"] == ["A"]
    assert result["explored_count"] == 1


# TEST 2: No Path Exists in Maze

def test_no_path_in_maze():

    grid = [
        ["S", "#", "G"]
    ]

    maze = Maze(grid)

    result = bfs(maze, maze.start, maze.goal)

    assert result["found"] == False
    assert result["path"] == []


# TEST 3: Goal is Unreachable in Graph

def test_unreachable_goal():

    graph = Graph()

    graph.add_edge("A", "B")
    graph.add_edge("C", "D")

    result = bfs(graph, "A", "D")

    assert result["found"] == False
    assert result["path"] == []


