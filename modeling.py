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
import wrangle as w




def get_scores():
    df = pd.read_csv('modeling_edu_data.csv')
    train, validate,test = w.split(df)
    X_train = train.drop(columns='risk_cat')
    y_train = train.risk_cat
    X_validate= validate.drop(columns='risk_cat')
    y_validate = validate.risk_cat
    X_test = test.drop(columns='risk_cat')
    y_test = test.risk_cat
    train_pred_df = pd.DataFrame()
    validate_pred_df = pd.DataFrame()
    test_pred_df = pd.DataFrame()
    train_pred_df['actual'] = train.risk_cat
    validate_pred_df['actual'] = validate.risk_cat
    test_pred_df['actual'] = test.risk_cat
    train_pred_df['baseline'] = 1
    validate_pred_df['baseline'] = 1
    test_pred_df['baseline'] = 1
    
    
    

    clf = DecisionTreeClassifier(max_depth=5, random_state=666)
    clf.fit(X_train, y_train)
    train_pred_df['clf'] = clf.predict(X_train)
    validate_pred_df['clf'] = clf.predict(X_validate)
    
    
    knn = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    knn.fit(X_train, y_train)
    train_pred_df['knn'] = knn.predict(X_train)
    validate_pred_df['knn'] = knn.predict(X_validate)
    
    # from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(C=1, random_state=666, intercept_scaling=1, solver='lbfgs')
    logit.fit(X_train,y_train)
    train_pred_df['logistic'] = logit.predict(X_train)
    validate_pred_df['logistic'] = logit.predict(X_validate)
    test_pred_df['logistic'] = logit.predict(X_test)
    
    
    rf = RandomForestClassifier(random_state=666)
    rf.fit(X_train,y_train)
    train_pred_df['random_forest'] = rf.predict(X_train)
    validate_pred_df['random_forest'] = logit.predict(X_validate)
    
    print('Descion Tree')
    print('===========================================')
    print('Train Data')
    print('======================')
    print(classification_report(train_pred_df.actual, train_pred_df.clf))
    print('Validate Data')
    print('======================')
    print(classification_report(validate_pred_df.actual, validate_pred_df.clf))
    print('KNN')
    print('===========================================')
    print(classification_report(train_pred_df.actual, train_pred_df.knn))
    print('Logistic Regression')
    print('===========================================')
    print('Train Data')
    print('======================')
    print(classification_report(train_pred_df.actual, train_pred_df.logistic))
    print('Validate Data')
    print('======================')
    print(classification_report(validate_pred_df.actual, validate_pred_df.logistic))
    print('Test Data')
    print('======================')
    print(classification_report(test_pred_df.actual, test_pred_df.logistic))
    print('Random Forest Classifier')
    print('===========================================')
    print('Train Data')
    print('======================')
    print(classification_report(train_pred_df.actual, train_pred_df.random_forest))
    print('Validate Data')
    print('======================')
    print(classification_report(validate_pred_df.actual, validate_pred_df.random_forest))
    


