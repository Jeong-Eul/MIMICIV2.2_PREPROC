import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

path = '/Users/DAHS/Desktop/MIMICIV2.2_PREPROC/data/csv/'
cohort = '/Users/DAHS/Desktop/MIMICIV2.2_PREPROC/data/demo.csv'

class Integration_data():
    def __init__(self):
        self.categorical_encoding()
        
    def create_stay_id(self):
        data=pd.read_csv(cohort, index_col = 0)

        hids=data['stay_id'].unique()
        print("Total stay",len(hids))
        return data, hids
    
    def categorical_encoding(self):
        data, hids=self.create_stay_id()
        data=self.getdata(data, hids)
        return data
        
            
    def getdata(self, dataset, ids):
        df_list = []   
        for sample in tqdm(ids):
            dyn=pd.read_csv(path+str(sample)+'.0/dynamic_proc.csv',header=[0,1])
            stat = dataset[dataset['stay_id']==int(sample)]
            
            columns_to_copy = ['subject_id', 'stay_id', 'hadm_id', 'Age', 'gender', 'ethnicity']
            for column in columns_to_copy:
                dyn[(  'demographic',        column)] = stat[column].values[0] 
            df_list.append(dyn)
  
        df = pd.concat(df_list, axis = 0)
        
        print("total stay dataframe shape",df.shape)
        df.columns=df.columns.droplevel(0)
        #encoding categorical
        gen_encoder = LabelEncoder()
        eth_encoder = LabelEncoder()

        gen_encoder.fit(df['gender'])
        eth_encoder.fit(df['ethnicity'])
    
        df['gender']=gen_encoder.transform(df['gender'])
        df['ethnicity']=eth_encoder.transform(df['ethnicity'])
        
        df.to_csv('MIMICIV2.0_semi_preproc.csv')
        return df