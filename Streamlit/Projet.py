import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

# Dataframe
df = pd.read_csv('./documents/formation/projet/streamlit/df_chgm_clmt.csv')

# Menu
st.sidebar.image('./documents/formation/projet/streamlit/gl_wrmg_sidebar.png', use_column_width=True)

# Création des pages
st.sidebar.subheader('Menu')
pages = ['Introduction', 'Observations', 'Causes', 'Conséquences', 'Zoom sur l\'Europe', 'Conclusion']
page = st.sidebar.radio('', pages)

st.sidebar.markdown("--") 

st.sidebar.subheader('Promotion Continue - Data Analyst - Janvier 2023')

st.sidebar.markdown("--") 

st.sidebar.subheader('Membres du projet : \n- Jason Fraissinet \n- Mohamed Gloulou \n- Marie Letoret')

if page == pages[0]:
    image = Image.open('./documents/formation/projet/streamlit/img_header.jpg')
    st.image(image)
    
    st.dataframe(df.head())
    
elif page == pages[1]:
    
    # Evolution de la température
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    t = ax.set_title("Evolution des températures dans le monde de 1880 à 2022")
    text = [ax.text(1990, -0.1 ,'Température moyenne', fontsize=8), ax.text(1997,-0.17 ,'1951 - 1980', fontsize=8)] 
   
    ax = sns.lineplot(x='Year', y='Glob', data=df)
    ax = sns.lineplot(x='Year', y= 0, data = df, linestyle='dotted', color = 'black')
    ax.set(xlabel='Année', ylabel='Température globale') 
    
    st.pyplot(fig)
    
    # Comparaison sur trois zones
    df.plot(x = 'Year', y = ['24N-90N', '24S-24N', '90S-24S'], 
            style = ["b-", "g-", "m-"], 
            title = "Réchauffement climatique - Comparaison sur trois zones", 
            figsize=(12,4), linewidth = 0.8);

elif page == pages[2]:
    
    # Causes 
    st.header("Les causes du changement climatique")
    st.markdown("Les causes du réchauffement climatique sont diverses et variées :\n- l'effet de serre, \n- la combustion des combustibles, \n- la déforestation,  \n- l'agriculture intensive, \n- l'urbanisation et l'expansion urbaine, \n- les activités industrielles, \n- les changements d'utilisation des terres, \n- la température de l’eau")
    
    # GES
    st.header("Les émissions des gaz à effet de serre")
    st.subheader("Évolution au cours des années")
    st.markdown("A la suite d’une analyse du jeu de données, nous pouvons observer ci-dessous la tendance générale des émissions de gaz à effet de serre à partir de 1850.")
    
    # Dataframe GES
    df_ges = pd.read_csv('./documents/formation/projet/streamlit/df_ges.csv')
    
    # Graphique 1
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    t = ax.set_title("Evolution des émissions des gaz à effet de serre dans le monde de 1830 à 2021 \n en prenant en compte le potentiel de réchauffement global")
    ax = sns.lineplot(x='Year', y='Data', hue='Gas', data=df_ges)
    ax.set(xlabel='Année')
    ax.set(ylabel='Données (Tg CH4 /an)')

    st.pyplot(fig)
    
    st.markdown("L’axe des ordonnées représente le potentiel de réchauffement global (PRG) cumulé de chacun des gaz, c'est-à-dire l’impact total qu’a chacun de ces gaz sur le réchauffement climatique. \nComme sur les courbes des hausses de températures, l’augmentation des émissions des gaz à effet de serre s’accentue à la fin du XXe siècle.")
    st.markdown("De plus, il est à noter que la répartition des émissions atmosphériques de ces gaz est inégale. Le graphique ci-dessous présente la proportion de chacun dans le monde en 2021 (tout en prenant en compte le PRG).")
    
    # Graphique 2
       
    # Dataframe GES / Pie
    df_ges_pie = pd.read_csv('./documents/formation/projet/streamlit/df_ges_pie.csv')
    
    fig = plt.figure(figsize = (3, 3))
    ax = fig.add_subplot(111)
    
    t = ax.set_title("Répartition des émissions atmosphériques de gaz à effet de serre \n dans le monde en 2021")
    
    ax.pie(x = df_ges_pie.Data, 
           labels = ['CH4', 'CO2', 'N2O'], 
           explode=(0, 0.05, 0),
           colors = ['red', 'orange', 'yellow'],
           shadow=True,
           autopct = lambda x: str(round(x, 2)) + '%', 
           pctdistance = 0.7, 
           labeldistance = 1.2)
    ax.axis('equal')
    st.pyplot(fig)
    
    st.markdown("D'après nos analyses, le dioxyde de carbone est responsable à 76% des émissions totales des gaz à effet de serre dans le monde.")
    
    st.header("L'évolution des températures est-elle corrélée aux émissions des gaz à effet de serre ?")
    st.markdown("Les objectifs de cette partie sont : \n- d'analyser la relation entre ces variables, \n- de la modéliser afin d'en mesurer la pertinence.")
    
    st.markdown("La réalisation d’une carte de chaleur nous a tout de suite permis de mettre en évidence les relations entre ces variables, et de noter les degrés de corrélations entre les différentes variables du jeu de données.")

    
    
    
    
    
    
    
    