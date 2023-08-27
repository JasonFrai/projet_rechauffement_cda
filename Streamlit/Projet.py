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
    st.markdown("Les causes du réchauffement climatique sont diverses et variées :\n- l'effet de serre, \n- la combustion des combustibles, \n- la déforestation,  \n- l'agriculture intensive, \n- l'urbanisation et l'expansion urbaine, \n- les activités industrielles, \n- les changements d'utilisation des terres, \n- la température de l’eau.")
    
    # GES
    st.header("Les émissions des gaz à effet de serre")
    st.subheader("Évolution au cours des années")
    st.markdown("A la suite d’une analyse du jeu de données, nous pouvons observer ci-dessous la tendance générale des émissions des gaz à effet de serre à partir de 1830.")
    
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
       
    st.markdown("**Répartition des émissions atmosphériques de gaz à effet de serre \n dans le monde en 2021 :**")
       
    # Dataframe GES / Pie
    df_ges_pie = pd.read_csv('./documents/formation/projet/streamlit/df_ges_pie.csv')
    
    col1, col2, col3= st.columns([1,2,1])
    with col2:
        fig, ax = plt.subplots(figsize=(5,5))
        ax = plt.pie(x = df_ges_pie.Data,
                     labels = ['CH4', 'CO2', 'N2O'], 
                     explode=(0, 0.05, 0),
                     colors = ['red', 'orange', 'yellow'],
                     shadow=True,
                     autopct = lambda x: str(round(x, 2)) + '%', 
                     pctdistance = 0.7, 
                     labeldistance = 1.2)
        
        st.pyplot(fig)
    
    st.markdown("D'après les données, le dioxyde de carbone est responsable à 76% des émissions totales des gaz à effet de serre dans le monde.")
    
    st.header("L'évolution des températures est-elle corrélée aux émissions des gaz à effet de serre ?")
    st.markdown("Les objectifs de cette sous-partie sont : \n- d'analyser la relation entre ces variables, \n- de la modéliser afin d'en mesurer la pertinence.")
    
    st.markdown("La réalisation d’une carte de chaleur nous a tout de suite permis de mettre en évidence les relations entre ces variables, et de noter les degrés de corrélations entre les différentes variables du jeu de données.")

    # Heatmap
    
    df_h = pd.read_csv('./documents/formation/projet/streamlit/df_chgm_clmt.csv')
    
    cor = df_h.corr()
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig)
    
    from scipy.stats import pearsonr
    st.markdown("Nous pouvons retrouver ces valeurs grâce aux tests statistiques de Pearson.")
    
    options = st.multiselect('Paire de variables soumise au test de Pearson', ['Année', 'Température globale', 'CH4', 'CO2', 'N2O'])
    
    for index, option in enumerate(options): 
        if option == 'Année': 
            options[index]='Year'
        elif option == 'Température globale':
            options[index]='Glob'
        elif option == 'CO2':
            options[index]='CO[2]'
        elif option == 'N20':
            options[index]='N[2]*O'
        elif option == 'CH4':
            options[index]='CH[4]'
        else:
            break
   
    if len(options) < 2:
        st.markdown("*Sélectionnez au moins deux variables*")
    elif len(options) > 2:
        st.markdown("Attention à ne sélectionner que deux variables")
    else:
        pearson_res = pearsonr(df[options[0]], df[options[1]])
        st.write('Le coefficient de corrélation selon le test de Pearson est :', f'{pearson_res[0]:.4f}')
    
    st.markdown("Nous pouvons ainsi observer :\n- une **très forte** corrélation entre les émissions de gaz à effet de serre et les années.\n- une **forte** corrélation entre les émissions de gaz à effet de serre et les températures globales.\n- une **faible** corrélation entre les émissions de gaz à effet de serre et l'Antarctique.")
    
    st.markdown("Sachant que les émissions de dioxyde de carbone représentent les ¾ des émissions globales des gaz à effet de serre, nous avons associé l’évolution des températures avec celui des émissions de CO2 et observé très nettement la corrélation entre ces deux variables.")
    
    # Graphique CO2 / Températures
    
    df_temp_c02 = pd.read_csv('./documents/formation/projet/streamlit/df_temp_c02.csv')
    df_global = pd.read_csv('./documents/formation/projet/streamlit/df_global.csv')
    
    fig = plt.figure(figsize=(10,6)) 
    ax1 = fig.add_subplot(111)

    t = ax1.set_title("Evolution des températures et des émissions de CO2")

    sns.lineplot(x='Year', y='Glob', data= df_temp_c02, label="Température", ax = ax1)
    ax1.set_ylabel('Evolution de la température')
    ax1.set_xlim(1880, 2021)

    ax2 = ax1.twinx()   
    sns.lineplot(x='Year', y='Data', 
                 data= df_global.loc[df_global['Gas'] == 'CO[2]'], 
                 ci = None, color = 'orange', label="CO2", ax=ax2)
    ax2.set_ylabel('Evolution des émissions de CO2', labelpad = 20)

    ax1.set_xlabel('Année')

    ax1.legend_.remove()
    ax2.legend_.remove()
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)

    st.pyplot(fig)
    
    # Conclusion
    st.markdown("Plus les années passent, plus les émissions de CO2 et les températures augmentent.")
    
    # Régression
    
    st.subheader("Régression linéaire")
    
    st.markdown("Nous avons utilisé la régression linéaire pour observer les degrés de corrélation entre la température globale et les gaz à effet de serre.")
    st.markdown("Pour cela, nous avons réalisé un graphique en nuage de points.")
    
    from scipy import stats
    
    fig = plt.figure(figsize=(20,8))
    ax = plt.subplot(1, 3, 1)

    X1 = df['CO[2]']
    Y1 = df['Glob']

    slope, intercept, r_value, p_value, std_err = stats.linregress(X1, Y1)

    plt.plot(X1, Y1, 'o', label='données')
    plt.plot(X1, intercept + slope*X1, 'r', label='régression')

    plt.xlabel("CO2") 
    plt.ylabel("Température globale") 
    plt.legend()

    r2_CO2 = f'{r_value**2:.4f}'
    ax.text(60000, -0.4, r"$ r^2 $ = " + r2_CO2, fontsize=12)

    ax2 = plt.subplot(1, 3, 2)


    X2 = df['CH[4]']
    Y2 = df['Glob']

    slope, intercept, r_value, p_value, std_err = stats.linregress(X2, Y2)

    plt.plot(X2, Y2, 'o', label='données')
    plt.plot(X2, intercept + slope*X2, 'r', label='régression')

    plt.xlabel("CH4") 
    plt.ylabel("Température globale") 
    plt.legend()

    r2_CH4 = f'{r_value**2:.4f}'
    ax2.text(500, -0.4, r"$ r^2 $ = " + r2_CH4, fontsize=12)

    ax3 = plt.subplot(1, 3, 3)

    X3 = df['N[2]*O']
    Y3 = df['Glob']

    slope, intercept, r_value, p_value, std_err = stats.linregress(X3, Y3)


    plt.plot(X3, Y3, 'o', label='données')
    plt.plot(X3, intercept + slope*X3, 'r', label='régression')

    plt.xlabel("N2O") 
    plt.ylabel("Température globale") 
    plt.legend()
    
    r2_N2O = f'{r_value**2:.4f}'
    ax3.text(16, -0.4, r"$ r^2 $ = " + r2_N2O, fontsize=12)
    
    st.pyplot(fig)
    
    st.markdown("Ces résultats confirment bien l'étroite relation entre la hausse des températures et celle des émissions de CO2.")
    

elif page == pages[3]:  
    
    # Conséquences 
    st.header("Quelques conséquences liées au réchauffement climatique")
    st.markdown("Dans le cadre de l’étude de notre problématique, nous avons poursuivi nos analyses en observant certains effets du réchauffement climatique : \n- L’augmentation du nombre de catastrophes naturelles, \n- La montée du niveau de la mer.")
    
    # Catastrophes naturelles
    st.header("Augmentation du nombre de catastrophes naturelles")
    
    st.markdown("A COMPLETER")
    
    # Hausse du niveau de la mer
    
    st.header("Evolution du niveau de la mer")
    
    # Graphique
    df_slr = pd.read_csv('./documents/formation/projet/streamlit/df_slr.csv')
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    t = ax.set_title("Evolution du niveau de la mer dans le monde de 1880 à 2018")
    
    ax = sns.lineplot(x='Year:year', y='Global mean sea level (reconstruction, CSIRO):number', ci=None, data=df_slr, label="CSIRO")
    ax = sns.lineplot(x='Year:year', y='Global mean sea level (reconstruction, DMW):number', ci=None, data=df_slr, label="US")
    ax = sns.lineplot(x='Year:year', y='Global mean sea level (satellite altimeter, CMEMS) :number', ci=None, data=df_slr, label="CMEMS")

    ax = sns.lineplot(x='Year:year', y=0, data = df_slr, linestyle='dotted', color = 'black')

    ax.set_xlabel('Année')
    ax.set_ylabel('Niveau de la mer (mm)')
    
    st.pyplot(fig)
    
    # Explications
    st.markdown("Le graphique illustre la montée du niveau de la mer entre 1880 et 2018, en se basant sur trois sources :"
                "\n- la ligne bleue représente une reconstruction (par satellites et altimètres), de 1880 à 2013, réalisée par l'organisme gouvernemental australien pour la recherche scientifique (CSIRO - Commonwealth Scientific and Industrial Research Organisation)."
    "\n- la ligne orange illustre une reconstruction un peu plus récente, de 1900 à 2015, réalisée par l'Université de Siegen."
    "\n- la ligne verte présente des données obtenues par des satellites altimétriques, entre 1993 et 2018. Cette étude a été menée par le Service de Surveillance de l'Environnement Marin de Copernicus (CMEMS).")
    
    st.markdown("Ces trois graphiques illustrent bien l'élévation du niveau de la mer dans le monde depuis la fin du XIXe siècle")
    
    st.markdown("Selon la synthèse du GIEC publiée en 2021, le niveau de la mer a augmenté de **0,20 m** entre 1901 et 2018. La moitié de cette hausse étant observée après 1980.")
    
    st.markdown("Le rythme d'augmentation du niveau des océans est de : \n- **1,3 mm/an** entre 1901 et 1971, \n- **1,9 mm/an** entre 1971 et 2006, \n- **3,7 mm/an** entre 2006 et 2018.")
    
    st.markdown("Le rythme annuel, en 2020, est estimé à plus de **3,5 mm par an**.")
    
    # Corrélations
    
    st.header("Corrélations avec l’évolution des températures")
    
    # Heatmap
    st.markdown("Pour étudier les corrélations existantes entre l’évolution des températures et celle du niveau de la mer, nous avons réalisé dans un premier temps une carte de chaleur.")
    
    df_slr_cor = pd.read_csv('./documents/formation/projet/streamlit/df_slr_cor.csv')
    
    cor = df_slr_cor.corr() 
    fig, ax = plt.subplots(figsize=(20,20)) 
    sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig)
    
    st.markdown("Nous pouvons ainsi noter : \n- une très forte corrélation entre le niveau de la mer (données issues de trois sources différentes) et les années. \n- une forte corrélation entre le niveau de la mer et les températures globales.")
    
    # Relation entre les températures et le niveau de la mer
    st.markdown("En complément, l’observation en parallèle de l’évolution des températures avec celui du niveau de la mer permet de vérifier graphiquement la corrélation entre ces deux variables.")
    
    # Graphique
    
    fig = plt.figure(figsize=(10,6)) 
    ax1 = fig.add_subplot(111)

    sns.lineplot(x='Year', y='Glob', data= df_slr_cor, label="Température", ax = ax1)
    ax1.set_ylabel('Evolution de la température')
    ax1.set_xlim(1880, 2013)

    ax2 = ax1.twinx()   
    sns.lineplot(x='Year', y='Global mean sea level (CSIRO)', 
                 data= df_slr_cor, ci = None, color = 'orange', 
                 label="Niveau de la mer (CSIRO)", ax=ax2)
    ax2.set_ylabel('Evolution du niveau de la mer', labelpad = 20)

    ax1.set_xlabel('Année')

    ax1.legend_.remove()
    ax2.legend_.remove()
    
    fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
    
    st.pyplot(fig)

    st.markdown("Les données qui ont été utilisées pour représenter l'évolution du niveau de la mer proviennent de l'organisme gouvernemental australien pour la recherche scientifique (CSIRO), cela afin d’observer la tendance de la courbe sur un temps plus long (entre 1880 et 2013). Nous pouvons bien constater en effet la corrélation entre les deux variables : plus le temps passe, plus les températures et le niveau de la mer augmentent.")


    # Régression
    
    st.subheader("Régression linéaire")
    
    st.markdown("Enfin, pour visualiser cette relation, nous avons réalisé un graphique en nuage de points, puis une modélisation linéaire")
    
    from scipy import stats

    df_sel = pd.read_csv('./documents/formation/projet/streamlit/df_sel.csv')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    X = df_sel['Global mean sea level (CSIRO)']
    Y = df_sel['Glob']

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    plt.plot(X, Y, 'o', label='données')
    plt.plot(X, intercept + slope*X, 'r', label='régression')

    plt.xlabel("Niveau de la mer") 
    plt.ylabel("Température globale") 

    r2 = f'{r_value**2:.4f}'
    ax.text(-50, -0.4, r"$ r^2 $ = " + r2, fontsize=12)
    
    st.pyplot(fig)


    st.markdown("Ce graphique confirme de nouveau la relation étroite entre la hausse des températures et l’augmentation du niveau de la mer.")




