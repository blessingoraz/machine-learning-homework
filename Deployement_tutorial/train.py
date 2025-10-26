#!/usr/bin/env python
# coding: utf-8

# To save our model, we use pickle, a system import
import pickle

import pandas as pd
import numpy as np
import sklearn

from sklearn.pipeline import make_pipeline # What does this do
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

#Load data and return dataframe
def load_data():

    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)

    return df

def train_model(df):

    y_train = df.churn.values

    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )

    train_dict = df[categorical + numerical].to_dict(orient='records') # DictVectorizer turns the dataframe into a dictionary, which is one-hot encoding
    pipeline.fit(train_dict, y_train)

    return pipeline

def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)

    # print('model saved to', filename)
    print(f'model saved to {filename}')

df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)




