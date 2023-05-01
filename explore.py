# Imports
import scipy.stats as stats
import numpy as np
import pandas as pd
from pandasai import PandasAI
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import explore as e
import prepare as p
import warnings
warnings.filterwarnings("ignore")
import plotly.express
from sklearn.cluster import KMeans

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_corr_heatmap(train):
    '''
    This function will display a heatmap of the potential correlations between variables in 
    our dataset
    '''
    # get the correlation values
    corr_matrix = train.corr()
    # create a plot
    plt.figure(figsize=(10,10))
    # plot a heatmap of the correlations
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # add a title
    plt.title('Heat Map of Correlation')
    # display the plot
    plt.show()

def best_kmeans(data,k_max):
    '''
    EXAMPLE USEAGE:
    data = scaled_train[['alcohol', 'quality']]
    best_kmeans(data,k_max=10)

    This function will produce an elbow graph with clusters
    '''
    # create empty list variables to store results
    means = []
    inertia = []
    # cycle through our desired amount of k's
    for k in range(1, k_max):
        # create a KMeans object with current k
        kmeans = KMeans(n_clusters=k)
        # fit the kmeans object to our data
        kmeans.fit(data)
        # store the kmeans object in our means list
        means.append(k)
        # store the inertia for current k in the inertia list
        inertia.append(kmeans.inertia_)
        # create a figure
        fig =plt.subplots(figsize=(10,5))
        # plot the current k and inertia
        plt.plot(means,inertia, 'o-')
        # add axis labels
        plt.xlabel('means')
        plt.ylabel('inertia')
        # remove gridlines
        plt.grid(True)
        # display the plot
        plt.show()

def apply_kmeans(data,k):
    '''
    This function will create a clusters based on the given variables and data
    '''
    # create a kmeans object with k clusters
    kmeans = KMeans(n_clusters=k)
    # fit the kmeans object on our data
    kmeans.fit(data)
    # store the clustered data as a new column
    data[f'k_means_{k}'] = kmeans.labels_
    # return the modified dataset
    return data


def modeling_split(train,validate,test):
    modeling = ['parent_educ_bachelors_degree', 'parent_educ_high_school',
           'parent_educ_masters_degree', 'parent_educ_some_college',
           'parent_educ_some_high_school', 'final_score', 'risk_rating',
           'free_reduced_lunch', 'test_prep_completed']

    train = train[modeling]
    validate = validate[modeling]
    test = test[modeling]
    X_train = train.drop(columns= ['final_score', 'risk_rating'])
    y_train = train[['final_score', 'risk_rating']]

    X_validate = validate.drop(columns= ['final_score', 'risk_rating'])
    y_validate = validate[['final_score', 'risk_rating']]


    X_test = test.drop(columns= ['final_score', 'risk_rating'])
    y_test = test[['final_score', 'risk_rating']]
    
    
    

    return X_train, y_train, X_validate, y_validate, X_test, y_test



def get_parents_edu_plot(train):
    parents_edu = list()
    parents_edu.append('risk_rating')
    parents_edu.append('final_score')

    for cols in train.columns:
        if 'parent_educ' in cols:
            parents_edu.append(cols)
        else:
            pass

    # Calculate the number of rows and columns for the subplots grid
    num_plots = len(parents_edu)
    cols = 2
    rows = num_plots // cols if num_plots % cols == 0 else (num_plots // cols) + 1

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10 * rows))

    for idx, edu in enumerate(parents_edu):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        sns.boxplot(data=train, x=f'{edu}', y="final_score", ax=ax)
        ax.set_title(f'Box plot for {edu}')
        ax.set_xlabel(f'{edu}')
        ax.set_ylabel('final_score')

    # Adjust layout and remove any unused subplots
    fig.tight_layout()
    for r in range(rows):
        for c in range(cols):
            if r * cols + c >= num_plots:
                axes[r, c].remove()

    plt.show()
    
################################################################################################


import pandas as pd
from scipy import stats

def kruskal_wallis_test(df, alpha=0.05):
    """
    Perform the Kruskal-Wallis test on final scores grouped by parental education levels.

    :param df: DataFrame containing the data.
    :param parent_educ_columns: List of column names representing different parental education levels.
    :param alpha: The significance level for the test. Default is 0.05.
    :return: A string describing the result of the Kruskal-Wallis test.
    """
    
    parent_educ_columns = [
    "parent_educ_bachelors_degree",
    'parent_educ_high_school',
    "parent_educ_masters_degree",
    'parent_educ_some_college',
    'parent_educ_some_high_school']
        
    
    # Group the final scores based on parental education levels
    groups = [df[df[col] == 1]['final_score'] for col in parent_educ_columns]

    # Perform the Kruskal-Wallis test
    H_statistic, p_value_kruskal = stats.kruskal(*groups)

    print("Kruskal-Wallis test statistic:", H_statistic)
    print("Kruskal-Wallis test p-value:", p_value_kruskal)

    # Interpret the results based on the chosen alpha value
    if p_value_kruskal < alpha:
        return "There is a statistically significant difference in final scores among the different parental education groups."
    else:
        return "There is no statistically significant difference in final scores among the different parental education groups."

    #################################################################################################


def mann_whitney_u_test(train,free_reduced_lunch_column='free_reduced_lunch', final_score_column= 'final_score', alpha=0.05):
    """
    Perform the Mann-Whitney U test on final scores grouped by the free_reduced_lunch column.

    :param df: DataFrame containing the data.
    :param free_reduced_lunch_column: The column name indicating if a student has free or reduced lunch.
    :param final_score_column: The column name containing the final scores.
    :param alpha: The significance level for the test. Default is 0.05.
    :return: A string describing the result of the Mann-Whitney U test.
    """
    # Group final scores based on the free_reduced_lunch column
    free_reduced_students = df[df[free_reduced_lunch_column] == 1][final_score_column]
    standard_students = df[df[free_reduced_lunch_column] == 0][final_score_column]

    # Perform the Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(free_reduced_students, standard_students, alternative='two-sided')

    print("Mann-Whitney U test statistic:", u_statistic)
    print("Mann-Whitney U test p-value:", p_value)

    # Interpret the results based on the chosen alpha value
    if p_value < alpha:
        return "Reject the null hypothesis"
    else:
        return "Fail to reject the null hypothesis"




##########################################################################################################
def lunch_visual(train):
    free_reduced_students = train[train['free_reduced_lunch'] == 1]['final_score']
    standard_students = train[train['free_reduced_lunch'] == 0]['final_score']

    # Check for independence (assuming you have a valid dataset with independent groups)

    # Check for normality using Shapiro-Wilk test
    _, p_value_free_reduced = stats.shapiro(free_reduced_students)
    _, p_value_standard = stats.shapiro(standard_students)

    print("Shapiro-Wilk test p-value for free/reduced lunch students:", p_value_free_reduced)
    print("Shapiro-Wilk test p-value for standard lunch students:", p_value_standard)

    # Check for normality visually using histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(free_reduced_students, kde=True)
    plt.title("Histogram for Free/Reduced Lunch Students")
    plt.xlabel("Final Grade")

    plt.subplot(1, 2, 2)
    sns.histplot(standard_students, kde=True)
    plt.title("Histogram for Standard Lunch Students")
    plt.xlabel("Final Grade")
    plt.show()

    # Check for normality visually using Q-Q plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    stats.probplot(free_reduced_students, plot=plt)
    plt.title("Q-Q Plot for Free/Reduced Lunch Students")

    plt.subplot(1, 2, 2)
    stats.probplot(standard_students, plot=plt)
    plt.title("Q-Q Plot for Standard Lunch Students")
    plt.show()

    # Check for homogeneity of variances using Levene's test
    _, p_value_levene = stats.levene(free_reduced_students, standard_students)
    print("Levene's test p-value:", p_value_levene)
    
    
    ###########################################################################################################################################################################

def testing_lunch(df):
        # Assuming you have a DataFrame named 'data' with columns 'is_free_reduced_lunch' and 'final_score'
    free_reduced_students = df[df['free_reduced_lunch'] == 1]['final_score']
    standard_students = df[df['free_reduced_lunch'] == 0]['final_score']

    # Perform the Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(free_reduced_students, standard_students, alternative='two-sided')

    print("Mann-Whitney U test statistic:", u_statistic)
    print("Mann-Whitney U test p-value:", p_value)

    alpha = 0.05
    if p_value < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
