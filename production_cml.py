import os 
import pandas as pd
import pickle

def import_pickle():
    '''
    this will load the pickel file
    '''
    with open('model_pickle.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def import_form_responses():
    url = 'https://docs.google.com/spreadsheets/d/1EI0nK3E2EJkcy2g23vNrHBvGo0ExEO-RF4SRnSGdmo4/edit?usp=sharing'
    csv_export_url = url.replace('/edit?usp=sharing', '/export?format=csv')
    df = pd.read_csv(csv_export_url)
    return df


def format_form_responses(df):
    df.columns = ['time_stamp','student_id', 'has_college_degree', 'free_reduced_lunch', 'parents_married', 'is_first_child','nr_siblings', 'rides_bus']
    student_id = df['student_id']
    df = df[['has_college_degree', 'free_reduced_lunch', 'parents_married', 'is_first_child','nr_siblings', 'rides_bus']].copy()
    change_values = {'Yes': 1,
                'No': 0
                }
    df = df.replace(change_values)
    
    model = import_pickle()
    
    predicted_proabability = pd.DataFrame(model.predict_proba(df), columns=['proab_of_low_risk', 'proab_of_high_risk'])
    df = df.merge(right=predicted_proabability.proab_of_high_risk, how='left', left_index=True, right_index=True)
    df['student_id'] = student_id
    df = df[['student_id', 'has_college_degree', 'free_reduced_lunch', 'parents_married','is_first_child', 'nr_siblings', 'rides_bus', 'proab_of_high_risk']]
    return df
    
def get_results(df):
    for n in range(len(df)):
        print('Student ID  | Proabaility of High Risk')
        print('=========================================')
        print(f'{df.student_id[n]}        | {((round(df.proab_of_high_risk,2))* 100)[n]}')
        
def run_form_responses():
    model = import_pickle()
    df = format_form_responses(import_form_responses())
    get_results(format_form_responses(import_form_responses()))
    
    
def main_menu():
    response = (input('Did you want to run thethe report? (y/n) ').lower().strip())
    
    if response == 'y':
        run_form_responses()
    elif response == 'n':
        main_menu()
    else:
        print('sorry, please enter y or n')
        main_menu()

main_menu()
        
        
        
        
