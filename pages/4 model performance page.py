import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import shap
from streamlit_shap import st_shap
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

class CustomLabelEncoder(LabelEncoder):
    def _init_(self, mapping):
        super()._init_()
        self.classes_ = list(mapping.keys())
        self.mapping = mapping

    def transform(self, x):
        return [self.mapping.get(item, -1) for item in x]
    


def create_label_encoder(mapping):
    return CustomLabelEncoder(mapping)




st.set_page_config(
page_title="Career Outcome Prediction",
layout="wide",
initial_sidebar_state="expanded")

# Read
with open('pages/model.pkl', 'rb') as file:
    data = pickle.load(file)
    
with open("dataframe_all.pkl","rb") as file:
    dataframe_all = pickle.load(file)["dataframe"]

model = data["model"]
le_career_outcome = data["le_career_outcome"]
le_season_outcome = data["le_season_outcome"]
X_test = data["X_test"]
y_test = data["y_test"]
y_pred_test = data["y_pred_test"]
young_players = data["young_players"]
class_names = ['Out of the League', 'Roster', 'Rotation', 'Starter', 'All-Star', 'Elite']


st.write("# Model performance")

def plot_feature_importances(cls, df_model):
    importances = cls.feature_importances_
    indices = np.argsort(importances)
    features = df_model.columns
    
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    
    # Instead of plt.show(), we return the figure object
    return plt.gcf()


def plot_confusion_matrix(y_test, predictions):
    
    
    #Plot confusion matrix (see plot_confusion_matrix function)
    # Create a confusion matrix
    cm = confusion_matrix(y_test, predictions)
    # Plot confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ['Out of the League', 'Roster', 'Rotation', 'Starter', 'All-Star', 'Elite'])
    cm_display.plot(cmap=plt.cm.Reds, ax=None)  # You can customize ax if needed
    plt.grid(False)
    # Rotate xticks
    plt.xticks(rotation= 90)
    

    return plt.gcf()

def generate_classification_report_heatmap(y_test, predictions, class_names):
    # Generating classification report
    report = classification_report(y_test, predictions)
    
    # Convert the classification report to a pandas DataFrame
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:  # Exclude headers and footers
        row = line.split()
        if row:
            report_data.append(row)

    report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    report_df['Class'] = [class_names[i] for i in report_df.index]

    # Convert columns to numeric (except 'Class' and 'Support')
    numeric_cols = ['Precision', 'Recall', 'F1-Score']
    report_df[numeric_cols] = report_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Set 'Class' as the index
    report_df.set_index('Class', inplace=True)

    # Dropping the 'Support' column for the heatmap
    heatmap_data = report_df.drop('Support', axis=1)

    # Creating the heatmap
    plt.figure(figsize=(8, 4))
    heatmap = sns.heatmap(heatmap_data, annot=True, cmap='flare', fmt='.2f', linewidths=0.5, vmin=0)
    heatmap.set_title('Classification report')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45)

    return plt.gcf()

st.write("## Confussion matrix")
fig2 = plot_confusion_matrix(y_test, y_pred_test)
st.pyplot(fig2)

st.write("## Classification report")
fig3 = generate_classification_report_heatmap(y_test, y_pred_test, class_names)
st.pyplot(fig3)


st.write("""Above, you can see the confusion matrix, a crucial tool that visualizes the performance of the classification model. It showcases the classification results, highlighting the model's accuracy in predicting different classes. The matrix reveals the true positive, true negative, false positive, and false negative values, allowing a comprehensive assessment of the model's performance across various classes.""")


st.write("## Feature importances plot")
fig1 = plot_feature_importances(model, X_test)
st.pyplot(fig1)

st.write("""Above we can see the importance of each feature on the career outcome prediction. Here we can see that the most important features are:
         
- **Season Outcome:** This variable holds substantial significance, likely indicating the influence of a player's performance across different seasons on their career trajectory. It serves as a pivotal indicator of overall success or failure within a specific season.

- **Minutes Played (Mins):** The amount of time a player spends on the court appears to hold considerable importance. This metric often correlates with a player's involvement and impact during a game, showcasing its relevance in determining career outcomes.

- **Points Scored:** The scoring ability of a player, represented by the number of points accumulated, emerges as a vital factor. It showcases a player's offensive prowess and effectiveness in contributing to the team's success, evidently influencing their career path.

- **Season Number:** The chronological aspect of a player's career journey, denoted by the season number, appears to carry significance. This variable likely encapsulates the player's progress, development, and performance evolution across multiple seasons, impacting their overall career trajectory.


""")












####################################################################################################



#st.write("# Global Explainability")
#shap.initjs()
#explainer = shap.Explainer(model)
#shap_values = explainer(X_test)
#
#
#
#class_names = ['Out of the League', 'Roster', 'Rotation', 'Starter', 'All-Star', 'Elite']
#average_absolute_shap_values = pd.DataFrame(index= X_test.columns, columns=class_names)
#
#for class_ in range(6):
#    shap_values_class_i = shap_values[:, :, class_]
#    
#    average_absolute_shap_values[class_names[class_]]= abs(shap_values_class_i.values).mean(axis=0)
#    
#average_absolute_shap_values['sum'] = average_absolute_shap_values.sum(axis=1)
#average_absolute_shap_values = average_absolute_shap_values.sort_values(by='sum', ascending=True)
#average_absolute_shap_values = average_absolute_shap_values.drop('sum', axis=1)
#
#
#
#
#
#colors = ['#008BFB', '#3769E8', '#8F37BB', '#CD0095', '#F70069', '#2CDB6A']
#fig = go.Figure()
#
#for i, col in enumerate(average_absolute_shap_values.columns):
#    fig.add_trace(go.Bar(y=average_absolute_shap_values.index, x=average_absolute_shap_values[col], name=col, orientation='h', marker=dict(color=colors[i]) ))
#
#fig.update_layout(
#    barmode='stack',
#    title='Absolute mean contribution of each variable to each career outcome prediction',
#    xaxis=dict(title='mean(|SHAP value|)'),
#    height=800  # Adjust the height value as needed
#)
#
#fig.update_layout(legend=dict(title='Career outcomes'))
#st.plotly_chart(fig)
#
#
#
#fig3 = shap.summary_plot(shap_values[:,:,0], X_test, show=False)
#
#st.write("## Out of the League Summary Plot")
#st_shap(fig3, height=500)
#
#
#
#
#st.write("## Rotation Summary Plot")
#st_shap(shap.summary_plot(shap_values[:,:,2], X_test, show=False), height=500)
#
#
#
#
#st.write("## Starter Summary Plot")
#st_shap(shap.summary_plot(shap_values[:,:,3], X_test, show=False), height=500)
#
#
#
#st.write("## Elite Summary Plot")
#st_shap(shap.summary_plot(shap_values[:,:,5], X_test, show=False), height=500)
#
#
#
#