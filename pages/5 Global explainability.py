import streamlit as st
from PIL import Image

st.set_page_config(
page_title="Career Outcome Prediction",
layout="wide",
initial_sidebar_state="expanded")

st.write("# Global Explainability")
plotly_plot = Image.open('plots/plotly_plot.png')

st.image(plotly_plot)


st.write("""In this graph we can see how each variable contributes to assessing which career outcome a player will have.
First of all, we can see that the number of seasons played on the NBA (season_number) is the variable which has 
more impact on predicting the career path of a player. As more seasons played on the NBA more experience and more chances 
to be on an NBA team roster. As NBA teams want prospect players but also experienced ones that can handle the up's and down's 
of an 82 games season.

The second important feature in contribution is the draft pick, as usual more talented players are selected first. 
And more talented players usually tend to have better career paths.
Finally, another interesting feature is the fta, which seems to have a bigger impact on Elite career outcome""")



st.write("# Summary Plots")

st.write("## Out of the League Summary Plot")
out_of_the_league = Image.open('plots/out_of_the_league.png')
st.image(out_of_the_league)

st.write("""
         Our gradient boosting model learnt that Out of the league players have the following characteristics:
         - Tend to have a low number of seasons played on NBA
         - An small amount of minutes played
         - Low number of defensive rebounds, assists and steals
         - High number of turnovers
         """)




st.write("## Rotation Summary Plot")
rotation = Image.open('plots/rotation.png')
st.image(rotation)

st.write("""
         Our gradient boosting model learnt that Rotation players have the following characteristics:
         - An small number of games started
         - An small number of minutes
         - An small number of points and assists, but higher number of turnovers
         - But tend to have a higher number of games, so they tend to play a high number of games but 
         few time on each one of those games.
         - The number of fouls it is also normally low. That's expected, since if your time on the court is low, 
         low chances are of commit fouls
         """)



st.write("## Starter Summary Plot")
starter = Image.open('plots/starter.png')
st.image(starter)

st.write("""
         Our gradient boosting model learnt that Starter players have the following characteristics:
         - High number of seasons played on the NBA
         - High number of steals, minutes, assists, blocks
         - High number of fouls, maybe a consequence of the higher number of time on the court
         """)

st.write("## Elite Summary Plot")
elite = Image.open('plots/elite.png')
st.image(elite)

st.write("""
         Our gradient boosting model learnt that Elite players have the following characteristics:
         - High number of free throws attempts and made
         - High number of seasons played on the NBA
         - High number of blocks, games started
         - High number of rebounds, both offensive and defensive
         - Also the have a high number of turnovers but as they have more time on the court they have also more chances
         """)
