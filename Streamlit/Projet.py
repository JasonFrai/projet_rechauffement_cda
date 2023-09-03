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
    st.markdown("Dans le cadre de l’étude de notre problématique, nous avons poursuivi nos analyses en observant certains effets du réchauffement climatique : \n- La montée du niveau de la mer, \n- L’augmentation du nombre de catastrophes naturelles.")
    
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
    
    st.markdown("Enfin, pour visualiser cette relation, nous avons réalisé un graphique en nuage de points, puis une modélisation linéaire.")
    
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

    
    
    # Catastrophes naturelles
    st.header("Augmentation du nombre de catastrophes naturelles")
    
    st.markdown('Les catastrophes naturelles étudiées à travers notre projet sont : les inondations, les tempêtes (typhons, moussons, etc.), les glissements de terrain, les sécheresses et les températures extrêmes.')
    
    # Graphique
    df_cn = pd.read_csv('./documents/formation/projet/streamlit/df_cn.csv')
    
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)

    df_cn.groupby(['year']).count().plot(kind="bar", ax=ax)

    plt.xlabel('Année')
    plt.ylabel('Nombre de catastrophes naturelles')

    plt.title("Nombre de catastrophes naturelles par année de 1960 à 2018")

    ax.get_legend().remove()
    
    st.pyplot(fig)
    
    st.markdown('Le nombre de catastrophes naturelles augmente au fil des années, avec une accélération à partir des années 2000. On peut donc supposer un lien entre la hausse des températures et le nombre d\'évènements climatiques graves.')
    
    st.header('Corrélations avec l’évolution des températures')
    
    
    # Régression
    
    st.subheader("Régression linéaire")
    
    st.markdown("Pour visualiser la relation entre l'évolution des températures et le nombre de catastrophes naturelles, nous avons réalisé un graphique en nuage de points, puis une modélisation linéaire.")
    
    from scipy import stats

    df_cor_cn = pd.read_csv('./documents/formation/projet/streamlit/df_cor_cn.csv')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    X = df_cor_cn['nb_cn']
    Y = df_cor_cn['Glob']

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    plt.plot(X, Y, 'o', label='données')
    plt.plot(X, intercept + slope*X, 'r', label='régression')

    plt.xlabel("Niveau de la mer") 
    plt.ylabel("Température globale") 

    r2 = f'{r_value**2:.4f}'
    ax.text(250, -0.17, r"$ r^2 $ = " + r2, fontsize=12)
    
    st.pyplot(fig)


    st.markdown("Ce graphique confirme la relation étroite entre la hausse des températures et l’augmentation du nombre de catastrophes naturelles dans le monde.")

    elif page == pages[4]:  
    
    # Intro 
    st.header("Zoom sur une région du monde : l'Europe")
    st.markdown("Nous avons mis en avant plusieurs causes et conséquences de cette hausse des température à l'échelle mondiale. L'objectif est maintenant d'observer ces différences à l'échelle de l'Europe. Pour cela, on va s'interésser à   : \n- La hausse des températures, \n- L'évolution du nombre de catastrophes naturelles,\n- Les émissions de CO2,\n- La croissance industrielle")
    
    # Hausse des température
    st.header("Hausse des températures")

    st.markdown("Dans un premier temps, rappelons la hausse des températures à l'échelle de l'Europe. Pour cela, le jeu de données de la NASA est utilisé.")

    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\Temp_1900.png")
        st.image(image,width=250)
    with col2:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\Temp_1960.png")
        st.image(image,width=250)
    with col3:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\Temp_2020.png")
        st.image(image,width=250)

    st.markdown("Dans un premier temps, on regarde les différences de températures. Etant donné que les valeurs des différences de températures sont indexées selon les latitudes des pays, on obtient trois couleurs distinctes. Comme analysé précédemment, l’augmentation des températures est plus importante en se rapprochant du pôle Nord. Comme attendu, les différences de températures sont de plus en plus importantes.")
    # Evolution du nombre de catastrophe naturelle
    
    st.header("Evolution du nombre de catastrophe naturelle")

    st.markdown("Les hausses de températures ont un impact sur le nombre de catastrophes naturelles, comme les tornades, secheresse, innondations, ect.. A l'aide d'un jeu de données fourni par la NASA, on observe l'évolution du nombre de catastrophes naturelles en Europe. ")
    col1,col2 = st.columns([1,1])
    with col1:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\cata_199x.png")
        st.image(image,width=250)
    with col2:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\cata_201x.png")
        st.image(image,width=250)

    st.markdown("Les cartes ont été tracées sur une décennie de catastrophes naturelles, en ne prenant en compte que les sécheresses, tempêtes et inondations sévères. Les tremblements de terres et éruptions volcaniqueont par exemple été filtrés car ces évènements ne sont, de notre point de vue, pas liés au réchauffement climatique.") 

    st.markdown("Avec les événements restants, on remarque une augmentation du nombre de catastrophes naturelles, pouvant être un signe de la fragilisation de l'équilibre à cause de l’augmentation des températures.")
     # Evolution du nombre des émissions de CO2
    
    st.header("Evolution des émissions de CO2")

    st.markdown("Pour tracer ces cartes, le jeu de données utilisé précédemment pour l’analyse a été de nouveau exploité.")
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\CO2_1880.png")
        st.image(image,width=230)
    with col2:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\CO2_1940.png")
        st.image(image,width=230)
    with col3:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\CO2_2020.png")
        st.image(image,width=230)

    st.markdown("On voit sur ces trois cartes l’augmentation très nette des émissions de CO2 au fil des ans. C’est une observation cohérente au vu des analyses précédentes, où les émissions de CO2 étaient fortement corrélées à l’augmentation des températures globales")
     # Croissance industrielle
    
    st.header("Croissance industrielle")

    st.markdown("Pour tracer les deux cartes ci-dessous, on a utilisé la croissance de l’index de production industrielle.")
    col1,col2 = st.columns([1,1])
    with col1:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\Indus_1990.png")
        st.image(image,width=250)
    with col2:
        image = Image.open(r"C:\Users\jason\Pictures\Projet data\Indus_2020.png")
        st.image(image,width=250)
    
    st.markdown("Ici, on remarque qu’entre 1990 et 2020 l’Europe est plus sur une phase de décroissance industrielle. Même si l’industrie a pu jouer un rôle important sur la pollution de l’air et l’augmentation des températures, son influence est aujourd’hui moindre en Europe.")

    st.markdown("C’est la limite de cette analyse centrée sur l’Europe. Le réchauffement climatique est un phénomène mondial, et même si l’industrie prend une place moins importante dans les causes de ce phénomène en Europe, ce n’est pas forcément le cas sur le reste du globe.")
elif page == pages[5]:  
    
    # Intro 
    st.title("Conclusion")
    st.markdown("Nos analyses conduisent à une constatation d’un réchauffement climatique à travers le dernier siècle, et à une accélération de celui-ci au cours des quarante dernières années. La décennie 2011-2020 a été la plus chaude jamais enregistrée. ")
    st.markdown("Les émissions de gaz à effet de serre, et notamment celles du dioxyde de carbone, dues aux activités humaines ont réchauffé la planète à un rythme sans précédent.")
    st.markdown("Cela a de nombreuses conséquences (vagues de chaleur, précipitations extrêmes, sécheresses, fonte des glaciers, etc.) et conduit à une vulnérabilité des écosystèmes et de la population (accès à l’eau et à l'alimentation, santé, etc.).")
    st.header("Conséquence sur l'agriculture")
    st.markdown("Cela a également des conséquences significatives dans le domaine de l'agriculture. Les cycles saisonniers sont perturbés par des températures changeantes et des schémas météorologiques imprévisibles, entraînant une diminution des rendements des cultures. Les sécheresses plus fréquentes et intenses augmentent le stress hydrique, tandis que les températures plus élevées favorisent les ravageurs et les maladies agricoles. Les cultures sensibles à la chaleur, comme le blé, le maïs et le riz, voient leurs rendements réduits. Le déplacement géographique des cultures devient nécessaire. Les fluctuations des rendements entraînent une instabilité économique et des prix alimentaires variables. Pour relever ces défis, des pratiques agricoles durables et des politiques d'adaptation au climat sont essentielles.")
    st.markdown("On aura souhaité approfondir ce lien entre le réchauffement climatique et l'agriculture. Cependant, cela demandait beaucoup d'analayse suplémentaires afin de relier les deux, comme les précipitations, les jours d'ensoleillement, ect.. ")
    st.markdown("On pense néanmoins qu'une approche basée sur l'analyse de base de données pourrait être bénéfique pour determiner les cultures optimales selon les régions du monde.")
    st.markdown("")
    st.markdown("")
    image = Image.open(r"C:\Users\jason\Pictures\Projet data\rechauffement.jpg")
    st.image(image,width=600)
    st.caption(body = "crédit : Journal Contrepoints")
    
    
    
    



