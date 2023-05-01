# Imports
import scipy.stats as stats
import numpy as np
import pandas as pd
from pandasai import PandasAI
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import explore as e
import prepare
import warnings
warnings.filterwarnings("ignore")
import plotly.express
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



def modeling_split(train,validate,test):
    modeling = ['parent_educ_bachelors_degree', 'parent_educ_high_school',
           'parent_educ_masters_degree', 'parent_educ_some_college',
           'parent_educ_some_high_school', 'final_score', 'risk_rating',
           'free_reduced_lunch', 'test_prep_completed']

    train = train[modeling]
    validate = validate[modeling]
    test = test[modeling]
    X_train = train.drop(columns= ['final_score', 'risk_rating'])
    y_train = train[['risk_rating']]
    X_validate = validate.drop(columns= ['final_score', 'risk_rating'])
    y_validate = validate[['risk_rating']]


    X_test = test.drop(columns= ['final_score', 'risk_rating'])
    y_test = test[['risk_rating']]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_baseline(df):
    new_df = pd.DataFrame()
    new_df['baseline'] = 'low_risk' * len(df)
    y_train = new_df['actual'] = df['risk_rating']
    
    return classification_report(y_train,new_df.baseline)