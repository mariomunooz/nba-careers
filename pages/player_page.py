import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as tri
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
page_title="Data Exploration",
page_icon="📊",
layout="wide",
initial_sidebar_state="expanded")

#The title and text
st.title("Data Exploration 📊 ")
st.write("In this tab we can see the most relevant information that we can extract through the data from the visual analytics.")

def hex_plot2(proportions1, proportions2, labels):
    def plot(ax, proportions, labels):
        N = len(proportions)
        proportions = np.append(proportions, 1)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = np.append(np.sin(theta), 0)
        y = np.append(np.cos(theta), 0)
        triangles = [[N, i, (i + 1) % N] for i in range(N)]
        triang_backgr = tri.Triangulation(x, y, triangles)
        triang_foregr = tri.Triangulation(x * proportions, y * proportions, triangles)
        cmap = plt.cm.rainbow_r
        colors = np.linspace(0, 1, N + 1)
        
        ax.tripcolor(triang_backgr, colors, cmap=cmap, shading='gouraud', alpha=0.4)
        ax.tripcolor(triang_foregr, colors, cmap=cmap, shading='gouraud', alpha=0.8)
        ax.triplot(triang_backgr, color='white', lw=2)
        
        for label, color, xi, yi in zip(labels, colors, x, y):
            ax.text(xi * 1.05, yi * 1.05, label,
                    ha='left' if xi > 0.1 else 'right' if xi < -0.1 else 'center',
                    va='bottom' if yi > 0.1 else 'top' if yi < -0.1 else 'center')
        ax.axis('off')
        ax.set_aspect('equal')
    
    # Create the plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plot(axs[0], proportions1, labels)
    plot(axs[1], proportions2, labels)
    
    st.pyplot(fig)  # Adjusts subplots to avoid overlapping
    
    return fig

def display_image_from_url(player_data_df):
    pl_nbapersonid = player_data_df['nbapersonid'].iloc[0]
    try:
        response = requests.get(f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pl_nbapersonid}.png")
        img = Image.open(BytesIO(response.content))
        st.image(img, caption= f"{player_data_df['player'].iloc[0]}", width=300)
    except Exception as e:
        response = requests.get(f"https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png")
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=f"{player_data_df['player'].iloc[0]} (No image available)", width=300)




with open("dataframe_all.pkl","rb") as file:
    dataframe_all = pickle.load(file)["dataframe"]


#List of all possible names
names = dataframe_all["player"].unique()

# Text input for user input
user_input = st.text_input('Type a player name:', '')

# Filter player names based on user input
filtered_players = [player for player in names if user_input.lower() in player.lower()]

# Display the filtered player names as options
selected_player = st.selectbox('Select a player:', options=filtered_players)

#['nbapersonid', 'player', 'draftyear', 'draftpick', 'career_outcome', 'season', 'team_list', 'num_teams_played', 'games', 'games_start','mins', 'fgm', 'fga', 'fgp', 
# 'fgm2', 'fga2', 'fgp2', 'fgm3', 'fga3','fgp3', 'ftm', 'fta', 'ftp', 'efg', 'off_reb', 'def_reb', 'tot_reb','ast', 'steals', 
# 'blocks', 'tov', 'tot_fouls', 'points','season_outcome', 'season_num']

player_data = dataframe_all[dataframe_all["player"]==selected_player]

left_co, cent_co,last_co = st.columns(3)

with cent_co:
    display_image_from_url(player_data)
#Career averages hex plot
games_pl = player_data["games"].sum()
ppg_career = player_data["points"].sum() / games_pl
reb_career = player_data["tot_reb"].sum() / games_pl
ast_career = player_data["ast"].sum() / games_pl
ste_career = player_data["steals"].sum() / games_pl
blo_career = player_data["blocks"].sum() / games_pl
efg_career = player_data["efg"].mean()
player_av_stats = [ppg_career,reb_career,ast_career,ste_career,blo_career,efg_career]

league_max = [dataframe_all["points"].max()/82,dataframe_all["ast"].max()/82,dataframe_all["tot_reb"].max()/82,dataframe_all["steals"].max()/82,dataframe_all["blocks"].max()/82,dataframe_all["efg"].max()]
labels = ["ppg","astpg","rebpg","stpg","blpg","efg"]
proportions = [player_av_stats[i]/league_max[i] for i in range(len(league_max))]
#hex_plot2(proportions,labels)


#Career peak

ppg_peak_career = player_data["points"].max() / 82
reb_peak_career = player_data["tot_reb"].max() / 82
ast_peak_career = player_data["ast"].max() / 82
ste_peak_career = player_data["steals"].max() / 82
blo_peak_career = player_data["blocks"].max() / 82
efg_peak_career = player_data["efg"].max()
player_peak_stats = [ppg_peak_career,reb_peak_career,ast_peak_career,ste_peak_career,blo_peak_career,efg_peak_career]

proportions_peak = [player_peak_stats[i]/league_max[i] for i in range(len(league_max))]
hex_plot2(proportions, proportions_peak,labels)





#Shooting splits

fgp = np.array(player_data["fgp"])
fgp3 = np.array(player_data["fgp3"])
ftp = np.array(player_data["ftp"])
years = np.array(player_data["season"])

mean_perc = dataframe_all[["season","fgp","fgp3","ftp"]].groupby(["season"]).mean()
mean_perc_y = mean_perc.loc[years]

Y = dataframe_all[["season","fga2","fga3","fta"]].groupby(["season"]).mean().loc[years]

filtered_individuals_2 = pd.DataFrame()
filtered_individuals_3 = pd.DataFrame()
filtered_individuals_t = pd.DataFrame()

for year in Y.index:
    y_dat = dataframe_all[dataframe_all["season"]==year]
    
    higher_values_2 = y_dat[y_dat["fga2"] > Y.loc[year, 'fga2']]
    higher_values_3 = y_dat[y_dat["fga3"] > Y.loc[year, 'fga3']]
    higher_values_t = y_dat[y_dat["fta"] > Y.loc[year, 'fta']]
    
    filtered_individuals_2 = pd.concat([filtered_individuals_2, higher_values_2])
    filtered_individuals_3 = pd.concat([filtered_individuals_3, higher_values_3])
    filtered_individuals_t = pd.concat([filtered_individuals_t, higher_values_t])

mean_perc_2 = filtered_individuals_2[["season","fgp"]].groupby(["season"]).mean()
mean_perc_3 = filtered_individuals_3[["season","fgp3"]].groupby(["season"]).mean()
mean_perc_t = filtered_individuals_t[["season","ftp"]].groupby(["season"]).mean()


fig, axs = plt.subplots(1, 3, figsize=(20, 6))
axs[0].plot(years, fgp, marker='o', label='Field Goal %')
axs[0].plot(years, np.array(mean_perc_y["fgp"]), label='Mean Field Goal %', linestyle='--')
axs[0].plot(years, np.array(mean_perc_2["fgp"]), label='Best shooters Mean Field Goal %', linestyle='dotted')
axs[0].set_title('Field goal percentage comparisson')
axs[0].set_xlabel('Year')
axs[0].set_ylabel('Percentage')
axs[0].legend()

axs[1].plot(years, fgp3, marker='o', label='3pt %')
axs[1].plot(years, np.array(mean_perc_y["fgp3"]), label='Mean 3pt %', linestyle='--')
axs[1].plot(years, np.array(mean_perc_3["fgp3"]), label='Best shooters mean 3pt %', linestyle='dotted')
axs[1].set_title('3pt percentage comparisson')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Percentage')
axs[1].legend()

axs[2].plot(years, ftp, marker='o', label='Free-throw %')
axs[2].plot(years, np.array(mean_perc_y["ftp"]), label='Mean Free throw %', linestyle='--')
axs[2].plot(years, np.array(mean_perc_t["ftp"]), label='Mean Field Goal %', linestyle='dotted')
axs[2].set_title('Free throw percentage comparisson')
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Percentage')
axs[2].legend()

st.pyplot(fig)



# Stats

if st.button("Show season stats:"):
    st.write(player_data)
