import streamlit as st
from PIL import Image

st.set_page_config(
page_title="Welcome",
page_icon=":basketball:",
layout="wide",
initial_sidebar_state="expanded")

#The title
st.title("Talent detection on NBA :basketball:")

#The text
st.write("""Detecting talent at early ages it is so important on the NBA, so we’ve developed this tool to help to understand why different players have different career paths.""")
st.write("# Data Collection")

st.write("""To compile this data, we utilized the [nba-api](https://github.com/swar/nba_api) Python package by Swar Patel, complementing it with 
         additional details sourced from [Basketball Reference](https://www.basketball-reference.com/). We employed [Wikidata](https://query.wikidata.org/) to map NBA.com player IDs 
         to their corresponding Basketball-Reference player IDs. This amalgamation of diverse data sources and 
         tools enabled the creation of two comprehensive datasets crucial for predicting NBA player careers 
         based on their initial seasons. """)

st.write("""So, we get the following data:
         \n- Player statistics
         \n- NBA player awards 
         
         \n\nMore details on [Muñoz Serrano, M. (2023). Forecasting NBA Careers. How to get NBA Data?](https://medium.com/@mariomunozserrano/forecasting-nba-careers-how-to-get-nba-data-3adedaa8984e) """)


st.write("# Methodology")

st.write("""
         With that we computed on a season level the season outcome of each player.
        \n**Season Outcome**
        \n- **Elite**: A player that won any All NBA award (1st, 2nd, or 3rd team), MVP, or DPOY in that season.
        \n- **All-Star**: A player selected to be an All-Star that season.
        \n- **Starter**: A player that started in at least 41 games in the season OR played at least 2000 minutes in the season.
        \n- **Rotation**: A player that played at least 1000 minutes in the season.
        \n- **Roster**: A player that played at least 1 minute for an NBA team but did not meet any of the above criteria.
        \n- **Out of the League**: A player that is not in the NBA in that season.
""")

st.write("""And with those season outcomes we can compute the career outcome of each player in the following way
         \n**Career outcome**
         \nHighest level of success that the player achieved for at least two seasons after his first four seasons in the league""")








