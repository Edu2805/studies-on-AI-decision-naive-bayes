import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('../CursoIA/CursoIA/3.Algoritmos de Machine Learning/insurance.csv', keep_default_na=False)
pd.set_option('display.max_columns', None)
base = base.drop(columns=['Unnamed: 0'])
#print(base.head())
#print(base.shape)
#print(base.isnull().sum())

y = base.iloc[:,7].values
x = base.drop(base.columns[7], axis=1).values
#print(x)

labelEncoder = LabelEncoder()
for i in range(x.shape[1]):
    if x[:,i].dtype == 'object':
        x[:,i] = labelEncoder.fit_transform(x[:,i])
#print(x)

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.3, random_state=12)

modelo = GaussianNB()
modelo.fit(x_treinamento, y_treinamento)

previsoes = modelo.predict(x_teste)

#print(previsoes)

accuracy = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')
#print(f'Acurácia: {accuracy}, Precisão: {precision}, Recall: {recall}, F1: {f1}')

report = classification_report(y_teste, previsoes)
print(report)