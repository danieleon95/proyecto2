#!/usr/bin/python
import pandas as pd
from sklearn.externals import joblib
import sys
import os

def predict(year,mileage,state,make,model):
    mod = joblib.load(os.path.dirname(__file__) + '/RandomForestPrice2.pkl') 
    cols = joblib.load(os.path.dirname(__file__) + '/columns_mod.pkl')
    d = {'Year':[year],'Mileage':[mileage],'State':[state],'Make':[make],'Model':[model]}
    df = pd.DataFrame(data=d)
    df = pd.get_dummies(df)
    df2 = pd.DataFrame(columns=cols)
    df2 = df2.append(df)
    df2 = df2.fillna(0)
    p1 = mod.predict(df2)
    return p1

predict(2010,3000,'OK',"BMW","1")

        