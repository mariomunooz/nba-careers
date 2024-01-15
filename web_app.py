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

st.write("""
         So, we defined 5 types of career outcomes depending how good was the career of an NBA player:
         - Out of the League
         - Roster
         - Rotation
         - Starter
         - All-Star
         - Elite
         """)

st.markdown('<a href="#Methodology">More details on Methodology</a>', unsafe_allow_html=True)

st.write("""
         And the following pages:
         - League page: Where you can see the distribution of different statistics, compare them, see each distribution by career outcome and the league leaders of each statistic selected
         - Player page: Where you can find a player and see different visualizations that help you to understand how good was the career path of the player selected
         - Model Performance page: Here we used two models to predict the career outcome of a player Random Forest and XGBoost as XGBoost performed better, here you can see different visualizations to understand the performance of the XGBoost model.
         - Global Explainability: Here you can see different shap visualisations to understand the patterns found by our XGBoost model
         """)






# Section with id 'hola_page'
st.markdown('<div id="Methodology"></div>', unsafe_allow_html=True)
st.header("Methodology")

st.write("""
         Using player NBA statistics and NBA player awards from all players drafted before 1983 we computed for each player and each season, different season outcomes that tells us how good was that season in the following way.
        \n**Season Outcome**
        \n- **Elite**: A player that won any All NBA award (1st, 2nd, or 3rd team), MVP, or DPOY in that season.
        \n- **All-Star**: A player selected to be an All-Star that season.
        \n- **Starter**: A player that started in at least 41 games in the season OR played at least 2000 minutes in the season.
        \n- **Rotation**: A player that played at least 1000 minutes in the season.
        \n- **Roster**: A player that played at least 1 minute for an NBA team but did not meet any of the above criteria.
        \n- **Out of the League**: A player that is not in the NBA in that season.
""")

st.write("""And with those season outcomes we computed the career outcome of each player in the following way
         \n**Career outcome**
         \nHighest level of success that the player achieved for at least two seasons after his first four seasons in the league""")

