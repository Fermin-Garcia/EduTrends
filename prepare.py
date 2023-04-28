# Import necessary libraries
from sklearn.impute import SimpleImputer
import acquire as a
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import explore as e

def prepare_edu():
    # Acquire the education data
    df = a.acquire_edu_data()
    # Clean the column names: set them to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # Drop columns containing 'unnamed' in their name
    df.drop(columns=[c for c in df.columns if 'unnamed' in c], inplace=True)
    # Drop the 'ethnicgroup' column
    df.drop(columns='ethnicgroup', inplace=True)
    # Create an empty list to store column names with missing values
    has_null = []

    # Iterate through the columns of the DataFrame and add columns with missing values to the list
    for cols in df.columns:
        if df[cols].isna().sum() > 0:
            has_null.append(cols)

    # Create a SimpleImputer instance with the most_frequent strategy
    imputer = SimpleImputer(strategy='most_frequent')

    # Fill the missing values in each column using the imputer
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))

    # Rename the columns
    df.columns = ['is_male', 'parent_educ', 'free_reduced_lunch', 'test_prep_completed', 'parent_marital_status',
                  'practicesport', 'is_first_child', 'nrsiblings', 'rides_bus',
                  'wkly_study_hours', 'math_score', 'reading_score', 'writing_score']

    # Create a dictionary to map values to new values
    value_change = {
        'female': 0,
        'male': 1,
        'no': 0,
        'yes': 1,
        'school_bus': 1,
        'private': 0,
        'none': 0,
        'completed': 1,
        'sometimes': 1,
        'regularly': 1,
        'never': 0,
        'free/reduced': 1,
        'standard': 0
    }

    # Replace values in the DataFrame using the dictionary
    df.replace(to_replace=value_change, inplace=True)

    # Calculate the mean of the scores and round it to 2 decimal places
    scores = ['writing_score', 'reading_score', 'math_score']
    
    # make who numbers with grade systems
    df['final_score'] = round(df[scores].mean(axis=1), 0).astype(int)
    df['writing_score'] = round(df['writing_score'],0).astype(int)
    df['reading_score'] = round(df['reading_score'],0).astype(int)
    df['math_score'] = round(df['math_score'],0).astype(int)
    
    # Create a list of object columns
    object_columns = list()
    for cols in df.columns:
        if df[cols].dtype == 'O':
            object_columns.append(cols)

    # Convert object columns to dummy variables and drop the first dummy column for each
    df = pd.get_dummies(df, columns=object_columns, drop_first=True)
    df.drop_duplicates(inplace=True)
    
    final_grade_cluster = e.apply_kmeans(data=df[['final_score']],k=3)
    df['risk_rating'] = final_grade_cluster['k_means_3']
    cluster_translation = {
    0 : 'low_risk',
    1 : 'high_risk',
    2 : 'at_risk'
    }
    
    df['risk_rating'] = df['risk_rating'].replace(cluster_translation)

    
    
    df.columns = df.columns.str.replace(' ', '_')
    
    return df


def split(df):
    """
    This function splits a dataframe into train, validate, and test in order to explore the data
    and to create and validate models. It takes in a dataframe and contains an integer for setting
    a seed for replication.
    Test is 20% of the original dataset. The remaining 80% of the dataset is divided between
    validate and train, with validate being .30*.80= 24% of the original dataset, and train being
    .70*.80= 56% of the original dataset.
    The function returns train, validate, and test data
	"""

    # Split the input dataframe into train and test sets, where test set is 20% of the original dataset
    train, test = train_test_split(df, test_size=0.2, random_state=123)

    # Further split the train set into train and validate sets,
    # where validate set is 24% (.3 * .8) of the original dataset
    train, validate = train_test_split(train, test_size=0.3, random_state=123)

    # Return train, validate, and test dataframes
    return train, validate, test
