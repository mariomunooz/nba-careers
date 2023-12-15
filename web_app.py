import streamlit as st
from PIL import Image

st.set_page_config(
page_title="Welcome",
page_icon=":basketball:",
layout="wide",
initial_sidebar_state="expanded")

#The title
st.title("NBA Analytics :basketball:")

#The subheader
st.subheader("Revolutionize the game through data")

#The text
st.write("As NBA revenues continue to increase, each season gains every time more significance. With rising revenues, teams are under amplified pressure to make strategic, impactful choices. This money incentive makes the competition stronger, making every action, like getting players, managing teams, or planning games, really important for a team to do well and for the whole league to work better.")
st.write("Therefore, teams are investing a lot of money in advisors and player scouts in order to build a winning team which can win at the short term, and, in consequence, generate financial benefit immediately. However, teams also need to look at the future, so apart from creating a strong team in the actuality, they must also take into account the future of their team, and of the other teams.")
st.write("With this objective, we offer teams a web page which can help coaches and team general managers to observe data about the players in the league in order to take decisions on how they build their teams.")

image = Image.open('revenues_ev.png')

st.image(image, caption='Revenues evolution through the years')


