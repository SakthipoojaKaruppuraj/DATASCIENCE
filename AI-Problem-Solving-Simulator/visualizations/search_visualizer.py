
import streamlit as st


def render_maze(
    maze,
    explored=None,
    frontier=None,
    current=None,
    path=None
):
    explored = set(explored or [])
    frontier = set(frontier or [])
    path = set(path or [])

    for r in range(maze.rows):

        columns = st.columns(maze.cols, gap="small")

        for c in range(maze.cols):

            position = (r, c)
            cell = maze.grid[r][c]

            background = "#e2e8f0"
            color = "#334155"
            label = "&nbsp;"

            if cell == "#":
                background = "#334155"
                color = "#ffffff"
                label = "#"

            elif cell == "S":
                background = "#16a34a"
                color = "#ffffff"
                label = "S"

            elif cell == "G":
                background = "#dc2626"
                color = "#ffffff"
                label = "G"

            elif position in path:
                background = "#2563eb"
                color = "#ffffff"
                label = "*"

            elif position == current:
                background = "#a855f7"
                color = "#ffffff"
                label = "●"

            elif position in explored:
                background = "#60a5fa"
                color = "#ffffff"
                label = "✓"

            elif position in frontier:
                background = "#f59e0b"
                color = "#ffffff"
                label = "○"

            with columns[c]:

                st.markdown(
                    f"""
                    <div style="
                        background-color: {background};
                        color: {color};
                        height: 48px;
                        width: 48px;
                        border-radius: 8px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 5px;
                    ">
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
