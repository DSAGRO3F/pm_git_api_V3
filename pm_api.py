import flask
from flask import Flask, jsonify, request, render_template, redirect, url_for, app, redirect
import pandas as pd
import numpy as np
import pickle
import sklearn
import json
from datetime import datetime, date
from dateutil import parser

# Initialisation Flask
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Définition des chemins
# 1. Chemin data en local:
# path_data = '/Users/olivierdebeyssac/Python_project_predictive_maintenance/data/'

# 2. Chemin data sur git
path_data = 'data/'


# Chemin modele en local
#path_model = '/Users/olivierdebeyssac/Python_project_predictive_maintenance/pm_git/pm_api/'
#f_model_name = 'best_model.pkl'
#model_path = path_model + f_model_name

# Chemin model sur git
model_path = 'best_model.pkl'


# Tentative de charger "df.pkl" sur Git. Les fichiers data sont trop importants, en chargeant juste "df", on se retrouve à dépasser la limite de 500MB sur Heroku.
# On ets contraint de procéder autrement. On va charger les deux fichiers ".csv" et faire dans l'api les traitements nécessaires
# Idée abandonnée. Autre solution: charger les fichiers "csv"

# 1. Chargement des data ".csv" en local
# nrows = 731 * 50
# path = '/Users/olivierdebeyssac/Python_project_predictive_maintenance/data'
# df_1 = pd.read_csv(path + '/' + 'equipment_failure_data_1.csv', sep=';')  # nrows = nrows
# df_2 = pd.read_csv(path + '/' + 'equipment_failure_data_2.csv', sep=';')

# 2. Noms fichiers data et chemins data
data_1 = 'equipment_failure_data_1.csv'
data_2 = 'equipment_failure_data_2.csv'

path_1 = path_data + data_1
path_2 = path_data + data_2

# 3. Chargement data
df_1 = pd.read_csv(path_1, sep=';')
df_2 = pd.read_csv(path_2, sep=';')


# 2. Concaténation des deux df
df = pd.concat([df_1, df_2], axis=0)
print("df: {}".format(df[0:2]))
print("df_shape: {}".format(df.shape))


# 3. Nombre de records par équipement (quelle est la fréquence de remontée data/eq.)
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%y')
print(df['DATE'][0:2])
print(df['DATE'].dtypes)
df_100001 = df[df['ID'] == 100001]
print("On a {} remontées de data par équipement".format(df_100001.shape[0]))


# 4. Filtrer les colonnes pour ne retenir que les colonnes correspondant aux capteurs.
filter_col = [col for col in df if col.startswith('S')]
print(filter_col)


# 5. Process de transformation, feature engineering.
# Principe:
# Pour un équipement donné, dans l'observation de son cycle de fonctionnement, on va voir apparaître à un moment donné des anomalies qui sont révélatrices de dysfonctionnements futurs.
# On veut se fixer une fenêtre d'observation qui puisse nous indiquer si ce genre d'anomaies se produisent, se sont produites ou ne se sont pas encore produites ==> "feature_window"
# Si les anomalies ne se sont pas encore produites on en déduit qu'aucune opération de maintenance n'est prévoir.
# Dans le cas contraire, on déduit qu'une opération de maintenance est à prévoir, à planifier.
feature_window = 21


# Création fonction qui crée une nouvelle colonne pour identifier un changement d'équipement dans la matrice.
def trigger_eq(df):
    df = df.sort_values(by=['ID', 'DATE'], ascending=[True, True])
    df['flipper'] = np.where((df['ID'] != df['ID'].shift(1)), 1, 0)
    return df


# Create new feature: 'TIME_SINCE_START' that gives time between equipment appears and current time.
dfx = trigger_eq(df)
dfx['DATE'] = dfx['DATE'].astype('str')
dfx['DATE'] = pd.to_datetime(dfx['DATE'], format='%d/%m/%y')

# Identify equipment change in our dataset and its associated set up date.
df_starter = dfx[dfx['flipper'] == 1]
df_starter = df_starter[['DATE', 'ID']]


# rename date to start_date
df_starter = df_starter.rename(index=str, columns={"DATE": "START_DATE"})
print(df_starter.info())


# Merge START_DATE to the original data set
dfx = dfx.sort_values(by=['ID', 'DATE'], ascending=[True, True])
df_starter = df_starter.sort_values(by=['ID'], ascending=[True])
dfx = dfx.merge(df_starter, on=['ID'], how='left')
print("dfx.head(3): {}".format(dfx.head(3)))


# Calcul du nombre de jours entre la date d'appartion de l'équipement and le jour actuel.
# Cette feature sera libellée "TIME_SINCE_START”.
# Création d'une nouvelle variable "too_soon". Lorsque cette variable=1, on a moins de 21 jours d'historique sur cet équipement.
# Pour chaque équipement, lorsque "too_soon" est différent de 1 (autrement dit des que l'équipement est en place depuis plus de 21 jours) on va calculer certaines valeurs spécifiques pour chaque capteur: moyenne, médiane, min, max


# calculate the number of days since the beginning of each well.
dfx['C'] = dfx['DATE'] - dfx['START_DATE']
dfx['TIME_SINCE_START'] = dfx['C'] / np.timedelta64(1, 'D')
dfx = dfx.drop(columns=['C'])
dfx['too_soon'] = np.where((dfx.TIME_SINCE_START < feature_window), 1, 0)
print(dfx[['TIME_SINCE_START', 'too_soon']][728:733])


# 6. Calcul des valeurs spécifiques.
l_suff = ['mean_val', 'med_val', 'min_val', 'max_val']


def spec_values(dfx):
    for col in filter_col:
        for suff in l_suff:
            col_name = col + '_' + suff

            if suff == 'mean_val':
                dfx[col_name] = np.where(dfx['too_soon'] == 0,
                                         dfx[col].rolling(window=feature_window, min_periods=1).mean(),
                                         dfx[col])

            if suff == 'med_val':
                dfx[col_name] = np.where(dfx['too_soon'] == 0,
                                         dfx[col].rolling(window=feature_window, min_periods=1).median(),
                                         dfx[col])

            if suff == 'min_val':
                dfx[col_name] = np.where(dfx['too_soon'] == 0,
                                         dfx[col].rolling(window=feature_window, min_periods=1).min(),
                                         dfx[col])

            else:
                dfx[col_name] = np.where(dfx['too_soon'] == 0,
                                         dfx[col].rolling(window=feature_window, min_periods=1).max(),
                                         dfx[col])
    return dfx


new_dfx = spec_values(dfx)
print("new_dfx.shape: {}".format(new_dfx.shape))
print('new_dfx: {}'.format(new_dfx[0:10]))


# 7. Calcul valeurs "peak"
def peak_val(new_dfx):
    for col in new_dfx.columns:
        l = col.split('_')
        if len(l) >= 3:

            if l[1] == 'mean':
                col_name = l[0] + '_peak'
                new_dfx[col_name] = np.where(new_dfx[col] == 0,
                                             new_dfx[col],
                                             new_dfx[l[0]] / new_dfx[col],
                                             )
    return new_dfx


df = peak_val(new_dfx)

# 8. Equilibrage du dataset.
# Commentaire:
# Pour que notre algorithme fonctionne correctement, il faut l'entrainer sur un set de valeurs qui soit équilibré, c'est à dire un data set pour lequel le nombre d'évènements où il y a défaillance de l'équipement soit à peu près similaire au nombre d'évènements où il n'y as défaillance.
# L'objectif ci dessous est d'équilibrer le data set.
# Dans cet objectif, on va considérer que la survenance de la panne qui a eu lieu à une date donnée aurait pu se produire dans les 28 jours qui ont précédé la panne.
# On va donc augmenter artificiellement le taux de panne.


# 1ère étape: équilibrage par extension de la fenètre d'obseravtion.
window_tgt = 28
df = df.sort_values(by=['ID', 'DATE'], ascending=[True, True])
df.reset_index(drop=True, inplace=True)
df_failure = df[df['EQUIPMENT_FAILURE'] == 1]
df_failure = df_failure[['DATE', 'ID']]
df_failure = df_failure.rename(columns={'DATE': 'FAILURE_DATE'})


df.sort_values(by='ID', inplace=True, ascending=True)
df_failure.sort_values(by='ID', inplace=True, ascending=True)


df_merged = df.merge(df_failure, on='ID', how='left')


df_merged['TIME_TO_FAILURE'] = df_merged['FAILURE_DATE'] - df_merged['START_DATE']
df_merged['TIME_TO_FAILURE'] = df_merged['TIME_TO_FAILURE'] / np.timedelta64(1, 'D')


# Check...
print("df['TIME_TO_FAILURE'][0:2]: {}".format(df_merged['TIME_TO_FAILURE'][0:2]))
print(df_merged.loc[1, ['ID', 'START_DATE', 'DATE', 'FAILURE_DATE', 'TIME_TO_FAILURE']])


df_merged['FAILURE_TGT'] = np.where((df_merged['TIME_TO_FAILURE'] < window_tgt) & (df_merged['TIME_TO_FAILURE'] >= 0), 1, 0)
# Comptage des valeurs 0 et 1:
print('Nombre de valeurs à 0: {}'.format(df_merged['FAILURE_TGT'].value_counts()[0] / len(df_merged)))
print('Nombre de valeurs à 1: {}'.format(df_merged['FAILURE_TGT'].value_counts()[1] / len(df_merged)))


# 2ème étape: utilisation de SMOTE pour meilleur équilibrage du dataset.
# on définit une méthode pour déterminer qu'une observation fera partie du train set, du validation ou du test set.
# On va créer un ensemble de 421 valeurs aléatoires comprises entre 0 et 10000.
# Chacune de ces valeurs est ensuite attribuée à un équipement.
# Puis, on décide que les ID qui portent les valeurs aléatoires <= 0.35 iront dans le train set, ceux pour lesquels on des valeurs aléatoires < 0.65 iront dans le validation set, les autres dans le test set.
# Ceci fait, on a alors un ensemble de mesures pour un équipement donné qui servira à l'entrainement et d'autres à la validation ou au test.
# Puis, en faisant un merge des df "pd_id" et "df", on obtient un df pour lequel tous les équipements sont tagés "Training", ou "Test".


# Get a Unique List of All IDs
df_temp = df_merged
df_id = df_temp.drop_duplicates(subset='ID')
df_id = df_id[['ID']]


np.random.seed(42)
df_id['random_val'] = (np.random.randint(0, 10000, df_id.shape[0])) / 10000
df_id = df_id[['ID', 'random_val']]


df_id['MODELING_GROUP'] = np.where((df_id.random_val <= 0.35), 'TRAINING',
                                   np.where((df_id.random_val <= 0.65), 'VALIDATION', 'TESTING'))
print(df_id.head(10))


tips_summed = df_id.groupby(['MODELING_GROUP'])['random_val'].count()
print(tips_summed)


# Attribution des tag à chaque équipement, merge des df.
df_merged = df_merged.sort_values(by=['ID'], ascending=[True])
df_id = df_id.sort_values(by=['ID'], ascending=[True])
new_df = df_merged.merge(df_id, on=['ID'], how='inner')


tips_summed_1 = new_df.groupby(['MODELING_GROUP'])['random_val'].count()
print(tips_summed_1)

# Méthode SMOTE pour équilibrage du dataset.
# Disparité des valeurs de défaillance et non défaillance.
print(new_df['EQUIPMENT_FAILURE'].value_counts())


# Features non nécessaires à l'entrainement des modèles mais nécessaires dans des phases ulérieures: les dates.
features = ['ID', 'MODELING_GROUP', 'EQUIPMENT_FAILURE',
            'DATE', 'START_DATE', 'TIME_SINCE_START',
            'FAILURE_DATE', 'TIME_TO_FAILURE', 'FAILURE_TGT',
            'S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18',
            'S8', 'AGE_OF_EQUIPMENT', 'S15_mean_val', 'S15_med_val',
            'S15_min_val', 'S15_max_val', 'S17_mean_val', 'S17_med_val',
            'S17_min_val', 'S17_max_val', 'S13_mean_val', 'S13_med_val',
            'S13_min_val', 'S13_max_val', 'S5_mean_val', 'S5_med_val',
            'S5_min_val', 'S5_max_val', 'S16_mean_val', 'S16_med_val',
            'S16_min_val', 'S16_max_val', 'S19_mean_val', 'S19_med_val',
            'S19_min_val', 'S19_max_val', 'S18_mean_val', 'S18_med_val',
            'S18_min_val', 'S18_max_val', 'S8_mean_val', 'S8_med_val',
            'S8_min_val', 'S8_max_val', 'S15_peak', 'S17_peak', 'S13_peak',
            'S5_peak', 'S16_peak', 'S19_peak', 'S18_peak', 'S8_peak']


# Features nécéssaires à l'entrainement des modèles.
training_features = ['ID', 'EQUIPMENT_FAILURE',
                     'TIME_SINCE_START', 'TIME_TO_FAILURE', 'FAILURE_TGT',
                     'S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18',
                     'S8', 'AGE_OF_EQUIPMENT', 'S15_mean_val', 'S15_med_val',
                     'S15_min_val', 'S15_max_val', 'S17_mean_val', 'S17_med_val',
                     'S17_min_val', 'S17_max_val', 'S13_mean_val', 'S13_med_val',
                     'S13_min_val', 'S13_max_val', 'S5_mean_val', 'S5_med_val',
                     'S5_min_val', 'S5_max_val', 'S16_mean_val', 'S16_med_val',
                     'S16_min_val', 'S16_max_val', 'S19_mean_val', 'S19_med_val',
                     'S19_min_val', 'S19_max_val', 'S18_mean_val', 'S18_med_val',
                     'S18_min_val', 'S18_max_val', 'S8_mean_val', 'S8_med_val',
                     'S8_min_val', 'S8_max_val', 'S15_peak', 'S17_peak', 'S13_peak',
                     'S5_peak', 'S16_peak', 'S19_peak', 'S18_peak', 'S8_peak']


# Construction des data sets.
df_features = new_df[features]
df_features = df_features.sort_values(by=['ID', 'DATE'], ascending=[True, True])
df_features = df_features[df_features['MODELING_GROUP'] != 'TRAINING']
print('df_features: {}'.format(df_features[0:10]))


df_test_val = df_features[training_features]
y_test_val = df_test_val['EQUIPMENT_FAILURE']
X_test_val = df_test_val.drop(columns=['EQUIPMENT_FAILURE'], axis=1)
print('X_test_val: {}'.format(X_test_val[0:10]))


# Prints
print('df_features.shape: {}'.format(df_features.shape))
print('df_test_val.shape: {}'.format(df_test_val.shape))
print("X_test_val.shape ==> {}".format(X_test_val.shape))
print('dimension y_test_val: {}'.format(y_test_val.shape))


# Construction fonction pour obtention dataframe df par app. dashbord
@app.route('/df', methods=['GET'])
def get_df():
    print('df: {}'.format(df_features[0:2]))
    print(df_features.columns)
    d_df = df_features.to_dict()
    json_format = jsonify(d_df)
    return json_format


# Définition des features pour analyse temporelle des capteurs
sensor_feat = ['ID', 'DATE', 'S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']


# Construction liste des Id équipements.
@app.route('/id', methods=['GET'])
def liste_id():
    ids = df_features.loc[:, 'ID'].values.tolist()
    unique_ids = np.unique(ids)
    unique_ids = unique_ids.tolist()
    return jsonify(unique_ids)


df_basic = df.loc[:, ['ID', 'S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']]


# Construction valeurs spécifiques capteurs
@app.route('/sensors_data/<id>', methods=['GET'])
def get_sensors_data(id):
    # print('valeur_id: {}'.format(id))
    df_features['ID'] = df_features['ID'].astype('int')

    # Sélection de l'équipement
    df_selected_eq = df_features[df_features['ID'] == int(id)]
    # print(df_selected_eq)

    # Pour chq capteur calculer rolling.mean()
    l_sensors = ['S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S5', 'S8']

    # Rolling function:
    roll_features = [(name + '_roll') for name in l_sensors]

    for i in range(len(roll_features)):
        feat_name = l_sensors[i]
        new_feat_name = roll_features[i]
        df_temp_1 = df_selected_eq.copy()
        df_temp_1[new_feat_name] = df_temp_1[feat_name].rolling(window=90).mean()
        df_selected_eq = df_temp_1

    # Conversion df en dict
    df_selected_eq = df_selected_eq.to_dict()

    return jsonify(df_selected_eq)


# Construction fonction qui renvoie X_test.
@app.route('/X_test/<id>', methods=['GET'])
def X_test_data(id):

    print('X_test_val.shape: {}'.format(X_test_val.shape))
    print('--------')
    print('df_features.shape: {}'.format(df_features.shape))
    print('--------')
    print("X_test_val ==> : {}".format(X_test_val[0:2]))
    print('--------')
    print('X_test_val.columns: {}'.format(X_test_val.columns))
    print('--------')
    X_test_val['ID'] = X_test_val['ID'].astype('int')
    # print(df['ID'].dtypes)

    # Relevant features for building X_test
    # training_features = ['ID',
    #                      'TIME_SINCE_START', 'TIME_TO_FAILURE', 'FAILURE_TGT',
    #                      'S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18',
    #                      'S8', 'AGE_OF_EQUIPMENT', 'S15_mean_val', 'S15_med_val',
    #                      'S15_min_val', 'S15_max_val', 'S17_mean_val', 'S17_med_val',
    #                      'S17_min_val', 'S17_max_val', 'S13_mean_val', 'S13_med_val',
    #                      'S13_min_val', 'S13_max_val', 'S5_mean_val', 'S5_med_val',
    #                      'S5_min_val', 'S5_max_val', 'S16_mean_val', 'S16_med_val',
    #                      'S16_min_val', 'S16_max_val', 'S19_mean_val', 'S19_med_val',
    #                      'S19_min_val', 'S19_max_val', 'S18_mean_val', 'S18_med_val',
    #                      'S18_min_val', 'S18_max_val', 'S8_mean_val', 'S8_med_val',
    #                      'S8_min_val', 'S8_max_val', 'S15_peak', 'S17_peak', 'S13_peak',
    #                      'S5_peak', 'S16_peak', 'S19_peak', 'S18_peak', 'S8_peak']
    # X_test = df[training_features]

    print(X_test_val.info())

    # Make predictions on X_Test_val.
    # 1. Load model.
    best_model = open(model_path, 'rb')
    best_model = pickle.load(best_model)

    # 2. Predictions.
    y_proba = best_model.predict_proba(X_test_val)
    y_pred = y_proba[:, 1]

    #3. Nouvelle variable dataframe
    df_proba = X_test_val.copy()

    # 3. Concaténer y_pred à X_test.
    df_proba['y_pred'] = y_pred
    print('------')
    print('df_proba.columns_1: {}'.format(df_proba.columns))
    print('------')

    # 4. Cut off
    cut_off = 0.5
    y_pred_cut_off = [0 if val < cut_off else 1 for val in y_pred]

    # 5. Concaténer y_pred_cut_off à X_test.
    df_proba['y_pred_cutoff'] = y_pred_cut_off
    print('df_proba.columns_2: {}'.format(df_proba.columns))

    # 6. Concaténer 'DATE' et X_test
    # df_sorted = X_test_val.sort_values(by=['ID'], ascending=True)
    dates = df_features['DATE'].to_list()

    # # Check DATE format
    # print('--------')
    # print('date format: {}'.format(df_features['DATE'].dtypes))
    # print('--------')
    # print('liste dates: {}'.format(dates[0:3]))
    # print('--------')
    df_proba['DATE'] = dates
    # print("X_test_val['DATE'][0:3]: {}".format(X_test_val['DATE'][0:3]))
    # print('--------')
    df_proba['DATE'] = df_proba['DATE'].astype(str)
    print('df_proba.columns_3: {}'.format(df_proba.columns))
    # X_test['DATE'] = X_test['DATE'].apply(lambda x: datetime.strptime(x,"%a %b %d %Y %H:%M:%S %Z%z (IST)"))

    # 4. Sélectionner l'ensemble des prédictions pour l'équipement "id"
    df_proba_selected_eq = df_proba[df_proba['ID'] == int(id)]
    print('df_proba_selected_eq.columns_1: {}'.format(df_proba_selected_eq.columns))
    print('------')
    print('df_proba_selected_eq.shape: {}'.format(df_proba_selected_eq.shape))

    # 5. Convertir en dictionnaire
    d_df_proba_eq = df_proba_selected_eq.to_dict()

    # 6. Sérialisation.
    json_df_proba_eq = jsonify(d_df_proba_eq)
    return json_df_proba_eq


app.run()

# get_sensors_data()

# get_df()
