import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
from scipy.stats import chi2_contingency
# Import our working functions
import wrangle as w

def get_parents_graph():
    df = pd.read_csv('preprocessed_edu_data.csv')
    drop_list = ['is_male', 'completed_test_prep', 'practiced_sport', 'wkly_study_hours_< 5', 'wkly_study_hours_> 10']
    df.drop(columns= drop_list, inplace= True)
    train, validate, test = w.split(df)
    college_degree_crosstab = pd.crosstab(train.has_college_degree, train.risk_cat)
    colors = ['green', 'red']
    college_degree_crosstab.plot(kind='bar',figsize=(8,8),color=colors)
    labels = ['Low Risk', 'High Risk'] # blue is low risk , orange is high risk
    plt.title("Charting the College Climb: Risk Categories by Parents' Education")
    plt.xlabel("Parents' Education")
    plt.ylabel("Count per Category")
    plt.xticks([0,1], ['No College Degree', 'Has College Degree'], )
    plt.legend(labels)
    plt.show()