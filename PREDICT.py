import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter import *
import tkinter as tk 
import matplotlib.pyplot as plt
Data = pd.read_csv('data.csv')

def stat(df):
    # Taille du dataset
    print("Taille du dataset: \n")
    print(df.shape,"\n")
    # Informations sur le datset
    print("Informations sur le datset: \n")
    print(df.info(),"\n")
    # Vérifier les valeurs manquantes
    print("les valeurs manquantes: \n")
    print(df.isnull().sum(),"\n")
    # Quelques stats sur le dataset
    print("stats sur le dataset: \n")
    print(df.describe(),"\n")
    # Avoir la distribution de la variable objectif
    print("la distribution de la variable objectif: (1: Malade , 0: Pas malade)\n")
    print(df['target'].value_counts(),"\n")


X = Data.drop(columns='target', axis=1)
Y = Data['target']

global X_test
global Y_test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)

def EvaluationDuModele():
    X_test_prediction = model.predict(X_test)
    accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy : ',accuracy)

def predict(T):
    input_data_as_numpy_array= np.asarray(T)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0]== 0):
        return 'La personne n\'est pas malade :)'
    else:
        return 'La personne est malade :('
        
def get_data():
    global data
    liste=tb.get("1.0","end-1c").split(",")   
    matrice = list(liste)
    for i in range(len(matrice)):
        matrice[i]=float(matrice[i])
    data= np.array(matrice).reshape(1, -1)
    data=data[0]
    data=tuple(data)

def change():
    aff = predict(data)
    label.configure(text=aff) 
    
my_w=tk.Tk()
my_w.title('Comparaison')
my_w.geometry("700x400")
titre=Label(my_w,text="Prediction",background='#fdf0d5',fg='#001219',font=("Javanese Text",40))
titre.pack()
my_w.configure(background='#fdf0d5')
tb=Text(my_w,height=0.5,width=27,background='white',fg='black')
tb.place(x=275,y=120)
b4=tk.Button(my_w,text="Confirmer",width=10,background='#780000',fg='white',font=("Helvetica",12,'bold'),command=lambda:get_data())
b4.place(x=330,y=170)
b2= tk.Button(my_w,text="Prédiction",width=10,background='#780000',fg='white',font=("Helvetica",12,'bold'),command=lambda:change())
b2.place(x=330,y=220)
label = tk.Label(my_w,background='#fdf0d5',fg='#001219',font=("Helvetica",12,'bold'))
label.place(x=235,y=280)
lb=Label(my_w,text="Inserez les données ",background='#fdf0d5',fg='#001219',font=("Javanese Text",17))
lb.place(x=70,y=102)
my_w.mainloop()

''' exemple d'utilisation: 1,1,1,1,1,1,1,1,1,1,1,1,1'''