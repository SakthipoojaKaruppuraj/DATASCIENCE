
import streamlit as st
import time

from environments.maze import Maze

from algorithms.search.bfs import bfs
from algorithms.search.dfs import dfs

from visualizations.search_visualizer import render_maze


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Problem-Solving Simulator",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------
# APPLICATION TITLE
# --------------------------------

st.title("🤖 AI Problem-Solving Simulator")

st.write(
    "Explore how different AI search algorithms "
    "solve problems step by step."
)


# --------------------------------
# SIDEBAR CONTROLS
# --------------------------------

st.sidebar.header("Search Configuration")

algorithm = st.sidebar.selectbox(
    "Select Search Algorithm",
    ["Breadth-First Search (BFS)", "Depth-First Search (DFS)"]
)

speed = st.sidebar.slider(
    "Animation Speed (seconds)",
    min_value=0.05,
    max_value=1.0,
    value=0.3,
    step=0.05
)


# --------------------------------
# MAZE ENVIRONMENT
# --------------------------------

grid = [
    ["S", ".", ".", "#", "."],
    ["#", "#", ".", "#", "."],
    [".", ".", ".", ".", "."],
    [".", "#", "#", "#", "."],
    [".", ".", ".", "G", "."]
]

maze = Maze(grid)


# --------------------------------
# ALGORITHM SELECTION
# --------------------------------

if algorithm == "Breadth-First Search (BFS)":

    search_function = bfs

    st.subheader("Breadth-First Search")

    st.caption(
        "BFS explores nodes level by level using a queue."
    )

else:

    search_function = dfs

    st.subheader("Depth-First Search")

    st.caption(
        "DFS explores one branch deeply using a stack."
    )


# --------------------------------
# DISPLAY MAZE
# --------------------------------

st.write("### Maze Environment")

render_maze(maze)


# --------------------------------
# RUN SEARCH
# --------------------------------

if st.button("▶ Run Search Animation", type="primary"):

    result = search_function(
        maze,
        maze.start,
        maze.goal
    )

    display = st.empty()

    st.write("### Search Execution")

    progress = st.progress(0)

    total_steps = len(result["steps"])

    for index, step in enumerate(result["steps"]):

        with display.container():

            st.write(
                f"Current Position: {step['current']}"
            )

            render_maze(
                maze,
                explored=step["explored"],
                frontier=step["frontier"],
                current=step["current"]
            )

            st.caption(
                f"Step {index + 1} of {total_steps}"
            )

        progress.progress(
            (index + 1) / total_steps
        )

        time.sleep(speed)


    # --------------------------------
    # FINAL RESULTS
    # --------------------------------

    st.write("### Final Search Path")

    render_maze(
        maze,
        path=result["path"]
    )

    if result["found"]:

        st.success("Goal reached successfully!")

    else:

        st.error("No path found!")


    # --------------------------------
    # SEARCH STATISTICS
    # --------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Explored Cells",
        result["explored_count"]
    )

    col2.metric(
        "Path Length",
        len(result["path"]) - 1 if result["found"] else 0
    )

    col3.metric(
        "Search Steps",
        len(result["steps"])
    )


    # --------------------------------
    # EXPLORATION ORDER
    # --------------------------------

    st.write("### Exploration Order")

    st.write(result["visited_order"])

    with st.expander("View Final Path Coordinates"):

        st.write(result["path"])
