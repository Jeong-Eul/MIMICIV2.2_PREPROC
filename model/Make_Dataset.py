import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class Integration_data():
    def __init__(self, get_data, first_try):
        self.categorical_encoding(get_data, first_try)
    def create_stay_id(self):
        data=pd.read_csv('/Users/DAHS/Desktop/circ_mimic_preprocessing_1day/data/demo.csv', index_col = 0)

        hids=data['stay_id'].unique()
        print("Total stay",len(hids))
        return hids
    
    def categorical_encoding(self, get_data, first_try):
        hids=self.create_stay_id()
        
        if (get_data == True)&(first_try == True):
            data=self.getdata(hids)
            
        else:
            data = pd.read_csv('Total.csv')
        
        return data
        
            
    def getdata(self,ids):
        df_list = []   
        data=pd.read_csv('/Users/DAHS/Desktop/circ_mimic_preprocessing_1day/data/demo.csv', index_col = 0)
        for sample in tqdm(ids):
            dyn=pd.read_csv('/Users/DAHS/Desktop/circ_mimic_preprocessing_1day/Data/csv/'+str(sample)+'.0/dynamic.csv',header=[0,1])
            stat = data[data['stay_id']==int(sample)]
            
            dyn.columns=dyn.columns.droplevel(0)
            columns_to_copy = ['subject_id', 'stay_id', 'hadm_id', 'Age', 'gender', 'ethnicity', 'insurance']
            for column in columns_to_copy:
                dyn[column] = stat[column].values[0]
                
            df_list.append(dyn)
            
        df = pd.concat(df_list, axis = 0)
        
        print("total stay dataframe shape",df.shape)
        
        #encoding categorical
        gen_encoder = LabelEncoder()
        eth_encoder = LabelEncoder()
        ins_encoder = LabelEncoder()

        gen_encoder.fit(df['gender'])
        eth_encoder.fit(df['ethnicity'])
        ins_encoder.fit(df['insurance'])

        df['gender']=gen_encoder.transform(df['gender'])
        df['ethnicity']=eth_encoder.transform(df['ethnicity'])
        df['insurance']=ins_encoder.transform(df['insurance'])
        
        df.to_csv('Total.csv')
        return df