import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class Integration_data():
    def __init__(self, get_data, first_try):
        self.categorical_encoding(get_data, first_try)
    def create_stay_id(self):
        labels=pd.read_csv('./data/csv/labels.csv', header=0)

        hids=labels.iloc[:,0].to_list()
        print("Total Samples",len(hids))
        return hids
    
    def categorical_encoding(self, get_data, first_try):
        hids=self.create_stay_id()
        
        if (get_data == True)&(first_try == True):
            data=self.getdata(hids)
            
        else:
            data = pd.read_csv('Total.csv')
            
        #encoding categorical
        gen_encoder = LabelEncoder()
        eth_encoder = LabelEncoder()
        ins_encoder = LabelEncoder()

        gen_encoder.fit(data['gender'])
        eth_encoder.fit(data['ethnicity'])
        ins_encoder.fit(data['insurance'])

        data['gender']=gen_encoder.transform(data['gender'])
        data['ethnicity']=eth_encoder.transform(data['ethnicity'])
        data['insurance']=ins_encoder.transform(data['insurance'])

        return data
        
            
    def getdata(self,ids):
        df_list = []   
        for sample in tqdm(ids):
            dyn=pd.read_csv('./data/csv/'+str(sample)+'/dynamic.csv',header=[0,1])
            stat=pd.read_csv('./data/csv/'+str(sample)+'/static.csv')
            
            dyn.columns=dyn.columns.droplevel(0)
            columns_to_copy = ['subject_id', 'stay_id', 'hadm_id', 'Age', 'gender', 'ethnicity', 'insurance']
            for column in columns_to_copy:
                dyn[column] = stat[column].values[0]
                
            df_list.append(dyn)
            
        df = pd.concat(df_list, axis = 0)
        
        print("total stay dataframe shape",df.shape)
        df.to_csv('Total.csv')
        return df