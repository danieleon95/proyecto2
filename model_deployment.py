#!/usr/bin/python
import pandas as pd
from sklearn.externals import joblib
import sys
import os

def predict(year,mileage,state,make,model):
    mod = joblib.load(os.path.dirname(__file__) + '/model.pkl') 
    cols = joblib.load(os.path.dirname(__file__) + '/columns_mod.pkl')
    d = {'Year':[year],'Mileage':[mileage],'State':[state],'Make':[make],'Model':[model]}
    df = pd.DataFrame(data=d)
    df = pd.get_dummies(df)
    df2 = pd.DataFrame(columns=cols)
    df2 = df2.append(df)
    df2 = df2.fillna(0)
    p1 = mod.predict(df2)
    return p1

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add information')
        
    else:

        p1 = predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        
        print('result: ', p1)

        
