'''
Variables:
---------

data : diabetes health indicators original dataset
X : features dataset
Y : target labels
pred : list of predicted labels
'''

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


def classifier(mat, model):
    '''
    Predict using the trained model

    Input:
    -----
        mat : NxM matrix
        model : the model choice
    Output:
    ------
        pred : list of predicted labels
    '''
    if model == 'SVM':
        model = pickle.load(open("svm_model.pkl", "rb"))
        pred = model.predict(mat)

    elif model == 'RF':
        model = pickle.load(open("classification_model.pkl", "rb"))
        pred = model.predict(mat)

    elif model == 'GBC':
        model = pickle.load(open("classification_model.pkl", "rb"))
        pred = model.predict(mat)

    else:
        raise Exception("Please select one of the three methods : SVM, RF, GBC")

    return pred


# Import data
data = pd.read_csv('data/validation_diabetes_health_indicators.csv')
data['Diabetes_012'] = data['Diabetes_012'].astype(int)

X = data.drop(columns=['Diabetes_012'])
y = data['Diabetes_012']



# Predict labels using trained models
models = ['SVM', 'RF', 'GBC']
for model in models:
    # Make prediction
    pred = classifier(X, model)

    # Evaluate model results
    accuracy = accuracy_score(pred, y)
    f1 = f1_score(pred, y, average='macro', zero_division=True)

    # Print results
    print(f'Model: {model}\n-----\nAccuracy: {accuracy:.2f} \nF1_score: {f1:.2f} \n')
