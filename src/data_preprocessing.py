import pandas as pd

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    if 'BMI Category' in df.columns:
        df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    if 'Blood Pressure' in df.columns:
        df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop(columns=['Blood Pressure'])
    if 'Person ID' in df.columns:
        df = df.drop(columns=['Person ID'])
    return df

def get_features_and_target(df):
    X = df.drop(columns=['Sleep Disorder', 'Quality of Sleep'])
    y = df['Sleep Disorder']
    return X, y