import os
import pandas as pd
import requests
from io import StringIO

def acquire_edu_data():
    # Check if the file exists in the current directory
    if os.path.exists('edu_analysis.csv'):
        # If it does, read the file using pandas
        df = pd.read_csv('edu_analysis.csv')
    else:
        # If it doesn't, download it from a Google Sheets URL
        url = 'https://docs.google.com/spreadsheets/d/19F4bcARhVG2auNxOD8FzrZ51FdtB8uXYI0Bj0al7SsI/edit?usp=sharing'
        csv_export_url = url.replace('/edit?usp=sharing', '/export?format=csv')

        # Use the requests library to download the CSV file
        response = requests.get(csv_export_url)
        content = response.content.decode('utf-8')

        # Read the content of the downloaded file with pandas
        df = pd.read_csv(StringIO(content))
    
    return df
