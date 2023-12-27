import os 
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
import os
import importlib
import warnings
import pdb
import gc

pd.set_option('mode.chained_assignment',  None) 
warnings.simplefilter(action='ignore', category=FutureWarning) 

local = '/Users/DAHS/Desktop/MIMICIV2.2_PREPROC/data'
root_dir = "/Users/DAHS/Desktop/early_prediction_of_circ_scl/"

def get_stay_id(target_df):
            
    stay = pd.read_csv(root_dir+"mimiciv/2.2"+"/icu/icustays.csv.gz")
    stay = stay[stay.notna()]
    
    target_df['charttime'] = pd.to_datetime(target_df['charttime'])

    stay['intime'] = pd.to_datetime(stay['intime'])
    stay['outtime'] = pd.to_datetime(stay['outtime'])


    target_df['stay_id'] = np.nan
    result = []

    unique_patient_ids = target_df['subject_id'].unique()

    for p in tqdm(range(len(unique_patient_ids))):
        
        p_id = unique_patient_ids[p]
        
        interest = target_df[target_df['subject_id']==p_id].copy().sort_values('charttime').reset_index(drop=True)
        stay_interest = stay[stay['subject_id']==p_id].copy()
        
        unique_stay_ids = stay_interest['stay_id'].unique()
        
        for s in  range(len(unique_stay_ids)):
            
            stay_id = unique_stay_ids[s]
            
            stay_interest2 = stay_interest[stay_interest['stay_id']==stay_id].copy()
            
            indices = np.where((interest['charttime'].values >= stay_interest2['intime'].values) & 
                            (interest['charttime'].values <= stay_interest2['outtime'].values))

            interest['stay_id'].loc[indices[0]] = stay_id

            result.append(interest)
            
    result_df = pd.concat(result)
    target_df = result_df[~(result_df['stay_id'].isnull())]
    return target_df

def generate_adm():
    data=pd.read_csv(local+"/cohort/cohort_icu_mortality_0_.csv.gz", compression='gzip', header=0, index_col=None)
    data['intime'] = pd.to_datetime(data['intime'])
    data['outtime'] = pd.to_datetime(data['outtime'])
    data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
    data['los']=data['los'].astype(str)
    data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
    data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
    data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
    data=data.drop(columns=['days', 'dummy','hours','min','sec'])
    data=data[data['los']>0]
    data['Age']=data['Age'].astype(int)

    print('[Complete generate admission]')
    
    return data

def generate_proc(data):
    proc=pd.read_csv(local+ "/features/preproc_proc_icu.csv.gz", compression='gzip', header=0, index_col=None)
    proc=proc[proc['stay_id'].isin(data['stay_id'])]
    proc[['start_days', 'dummy','start_hours']] = proc['event_time_from_admit'].str.split(' ', -1, expand=True)
    proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', -1, expand=True)
    proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
    proc[['start_days', 'dummy','start_hours']] = proc['stop_hours_from_admit'].str.split(' ', -1, expand=True)
    proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', -1, expand=True)
    proc['stop_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
    proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
    proc=proc[proc['start_time']>=0]
    
    ###Remove where event time is after discharge time
    proc=pd.merge(proc,data[['stay_id','los']],on='stay_id',how='left')
    proc['sanity']=proc['los']-proc['start_time']
    proc=proc[proc['sanity']>0]
    del proc['sanity']
    
    print('[Complete generate procedure]')
    
    return proc

def generate_microlabs(data):
    milabs=pd.read_csv(local+ "/features/preproc_microlabs.csv.gz", compression='gzip', header=0, index_col=None)
    milabs=milabs[milabs['hadm_id'].isin(data['hadm_id'])]
    milabs[['start_days', 'dummy','start_hours']] = milabs['lab_time_from_admit'].str.split(' ', -1, expand=True)
    milabs[['start_hours','min','sec']] = milabs['start_hours'].str.split(':', -1, expand=True)
    milabs['start_time']=pd.to_numeric(milabs['start_days'])*24+pd.to_numeric(milabs['start_hours'])
    milabs=milabs.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
    milabs=milabs[milabs['start_time']>=0]
    
    ###Remove where event time is after discharge time
    milabs=pd.merge(milabs,data[['hadm_id','los']],on='hadm_id',how='left')
    milabs['sanity']=milabs['los']-milabs['start_time']
    milabs=milabs[milabs['sanity']>0]
    del milabs['sanity']
    milabs = get_stay_id(milabs)
    
    print('[Complete generate microlab events]')
    
    return milabs

def generate_out(data):
    out=pd.read_csv(local+ "/features/preproc_out_icu.csv.gz", compression='gzip', header=0, index_col=None)
    out=out[out['stay_id'].isin(data['stay_id'])]
    out[['start_days', 'dummy','start_hours']] = out['event_time_from_admit'].str.split(' ', -1, expand=True)
    out[['start_hours','min','sec']] = out['start_hours'].str.split(':', -1, expand=True)
    out['start_time']=pd.to_numeric(out['start_days'])*24+pd.to_numeric(out['start_hours'])
    out=out.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
    out=out[out['start_time']>=0]
    
    ###Remove where event time is after discharge time
    out=pd.merge(out,data[['stay_id','los']],on='stay_id',how='left')
    out['sanity']=out['los']-out['start_time']
    out=out[out['sanity']>0]
    del out['sanity']
    
    print('[Complete generate output events]')
    
    return out

def generate_chart(data):
    chunksize = 5000000
    final=pd.DataFrame()
    for chart in tqdm(pd.read_csv(local+ "/features/preproc_chart_icu.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
        chart=chart[chart['stay_id'].isin(data['stay_id'])]
        chart[['start_days', 'dummy','start_hours']] = chart['event_time_from_admit'].str.split(' ', -1, expand=True)
        chart[['start_hours','min','sec']] = chart['start_hours'].str.split(':', -1, expand=True)
        chart['start_time']=pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
        chart=chart.drop(columns=['start_days', 'dummy','start_hours','min','sec','event_time_from_admit'])
        chart=chart[chart['start_time']>=0]

        ###Remove where event time is after discharge time
        chart=pd.merge(chart,data[['stay_id','los']],on='stay_id',how='left')
        chart['sanity']=chart['los']-chart['start_time']
        chart=chart[chart['sanity']>0]
        del chart['sanity']
        del chart['los']
        
        if final.empty:
            final=chart
        else:
            final=final.append(chart, ignore_index=True)
    
    print('[Complete generate chart events]')
    
    return final

def generate_labs(data):
    chunksize = 10000000
    final=pd.DataFrame()
    for labs in tqdm(pd.read_csv(local+ "/features/preproc_labs.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
        labs=labs[labs['hadm_id'].isin(data['hadm_id'])]
        labs[['start_days', 'dummy','start_hours']] = labs['lab_time_from_admit'].str.split(' ', -1, expand=True)
        labs[['start_hours','min','sec']] = labs['start_hours'].str.split(':', -1, expand=True)
        labs['start_time']=pd.to_numeric(labs['start_days'])*24+pd.to_numeric(labs['start_hours'])
        labs=labs.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        labs=labs[labs['start_time']>=0]

        ###Remove where event time is after discharge time
        labs=pd.merge(labs,data[['hadm_id','los']],on='hadm_id',how='left')
        labs['sanity']=labs['los']-labs['start_time']
        labs=labs[labs['sanity']>0]
        del labs['sanity']
        
        if final.empty:
            final=labs
        else:
            final=final.append(labs, ignore_index=True)
        final = get_stay_id(final)
        
    print('[Complete generate lab events]')
        
    return final


def generate_meds(data):
    meds=pd.read_csv(local+ "/features/preproc_med_icu.csv.gz", compression='gzip', header=0, index_col=None)
    meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', -1, expand=True)
    meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
    meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
    meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', -1, expand=True)
    meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
    meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
    meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
    #####Sanity check
    meds['sanity']=meds['stop_time']-meds['start_time']
    meds=meds[meds['sanity']>0]
    del meds['sanity']
    #####Select hadm_id as in main file
    meds=meds[meds['stay_id'].isin(data['stay_id'])]
    meds=pd.merge(meds,data[['stay_id','los']],on='stay_id',how='left')

    #####Remove where start time is after end of visit
    meds['sanity']=meds['los']-meds['start_time']
    meds=meds[meds['sanity']>0]
    del meds['sanity']
    ####Any stop_time after end of visit is set at end of visit
    meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
    del meds['los']
    
    meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
    meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
    
    
    print('[Complete generate medication events]')
    
    return meds


def generate_ing(data):
    ing=pd.read_csv(local+ "/features/preproc_ing_icu.csv.gz", compression='gzip', header=0, index_col=None)
    ing[['start_days', 'dummy','start_hours']] = ing['start_hours_from_admit'].str.split(' ', -1, expand=True)
    ing[['start_hours','min','sec']] = ing['start_hours'].str.split(':', -1, expand=True)
    ing['start_time']=pd.to_numeric(ing['start_days'])*24+pd.to_numeric(ing['start_hours'])
    ing[['start_days', 'dummy','start_hours']] = ing['stop_hours_from_admit'].str.split(' ', -1, expand=True)
    ing[['start_hours','min','sec']] = ing['start_hours'].str.split(':', -1, expand=True)
    ing['stop_time']=pd.to_numeric(ing['start_days'])*24+pd.to_numeric(ing['start_hours'])
    ing=ing.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
    #####Sanity check
    ing['sanity']=ing['stop_time']-ing['start_time']
    ing=ing[ing['sanity']>0]
    del ing['sanity']
    #####Select hadm_id as in main file
    ing=ing[ing['stay_id'].isin(data['stay_id'])]
    ing=pd.merge(ing,data[['stay_id','los']],on='stay_id',how='left')

    #####Remove where start time is after end of visit
    ing['sanity']=ing['los']-ing['start_time']
    ing=ing[ing['sanity']>0]
    del ing['sanity']
    ####Any stop_time after end of visit is set at end of visit
    ing.loc[ing['stop_time'] > ing['los'],'stop_time']=ing.loc[ing['stop_time'] > ing['los'],'los']
    del ing['los']
    
    ing['rate']=ing['rate'].apply(pd.to_numeric, errors='coerce')
    ing['amount']=ing['amount'].apply(pd.to_numeric, errors='coerce')
    
    print('[Complete generate ingredient events]')
    
    return ing


def tabularization(feat_med, feat_ing, feat_out, feat_chart, feat_lab, feat_micro, feat_proc,
                   final_meds, final_ing, final_proc, final_out, final_chart, final_labs, final_micro, valid_stay_ids, data):
    
    print("# Unique gender: ", data.gender.nunique())
    print("# Unique ethnicity: ", data.ethnicity.nunique())
    print("=====================")
    print('Number of patient: ', len(data.subject_id.unique()))
    print('Number of stay: ', len(data.stay_id.unique()))
    print('Expected value of observation: ', data['los'].sum())
    print("=====================")
    print()
    
    
    for hid in tqdm(valid_stay_ids, desc = 'Tabularize EHR for total stay 20,809'):
        gc.collect()
        grp=data[data['stay_id']==hid]
        los = int(grp['los'].values)
        if not os.path.exists(local+"/csv/"+str(hid)):
            os.makedirs(local+"/csv/"+str(hid))
        
        dyn_csv=pd.DataFrame()
        
        ###MEDS
        if(feat_med):
            feat=final_meds['itemid'].unique()
            feat_rate = [item + '_rate' for item in feat]
            df2=final_meds[final_meds['stay_id']==hid]
            if df2.shape[0]==0:
                dose=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                dose=dose.fillna(0)
                dose.columns=pd.MultiIndex.from_product([["MEDS"], dose.columns])
                
                rate=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat_rate)
                rate=rate.fillna(0)
                rate.columns=pd.MultiIndex.from_product([["RATE"], rate.columns])
            else:
                dose=df2.pivot_table(index='start_time',columns='itemid',values='amount')
                rate=df2.pivot_table(index='start_time',columns='itemid',values='rate')
                df2=df2.pivot_table(index='start_time',columns='itemid',values='stop_time') #value는 큰 의미 없음

                add_indices = pd.Index(range(los)).difference(df2.index) # 처방 된 시간과 los 비교 후 처방이 이루어지지 않은 시간 포인트만큼 시간 인덱스 생성
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan) 
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.ffill()
                df2=df2.fillna(0)

                dose=pd.concat([dose, add_df])
                dose=dose.sort_index()
                dose=dose.ffill()
                dose=dose.fillna(0)
                
                rate=pd.concat([rate, add_df])
                rate=rate.sort_index()
                rate=rate.ffill()
                rate=rate.fillna(0)
    
                df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0) #end time - start time
                df2[df2>0]=1 # 약물 처방 시간 만큼 1
                df2[df2<0]=0 #약을 처방 받지 않은 경우에는 0으로 채웠었기 때문에 sub 연산시 음 값을 가지게 되어 0으로 변환됨
        
                dose.iloc[:,0:]=df2.iloc[:,0:]*dose.iloc[:,0:] # 실제 처방 된 경우의 값만 살아 남음
                rate.iloc[:,0:]=df2.iloc[:,0:]*rate.iloc[:,0:]
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(dose.columns)))
                feat_df_rate=pd.DataFrame(columns=list(set(feat_rate)-set(rate.columns)))
                
                dose=pd.concat([dose,feat_df],axis=1)
                rate = pd.concat([rate, feat_df_rate], axis=1)

                dose=dose[feat]
                rate=rate[feat_rate]
                
                dose=dose.fillna(0)
                rate=rate.fillna(0)
                
                dose.columns=pd.MultiIndex.from_product([["MEDS"], dose.columns])
                rate.columns=pd.MultiIndex.from_product([["RATE"], rate.columns])
        
            if(dyn_csv.empty):
                dyn_csv= pd.concat([dose, rate],axis=1)
            else:
                medication = pd.concat([dose, rate],axis=1)
                dyn_csv=pd.concat([dyn_csv,medication],axis=1)
            
        
        ###INGS
        if(feat_ing):
            feat=final_ing['itemid'].unique()
            df2=final_ing[final_ing['stay_id']==hid]
            if df2.shape[0]==0:
                amount=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                amount=amount.fillna(0)
                amount.columns=pd.MultiIndex.from_product([["INGS"], amount.columns])
            else:
                amount=df2.pivot_table(index='start_time',columns='itemid',values='amount')
                df2=df2.pivot_table(index='start_time',columns='itemid',values='stop_time')
                
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.ffill()
                df2=df2.fillna(0)

                amount=pd.concat([amount, add_df])
                amount=amount.sort_index()
                amount=amount.ffill()
                amount=amount.fillna(0)
            
                df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                df2[df2>0]=1
                df2[df2<0]=0
                amount.iloc[:,0:]=df2.iloc[:,0:]*amount.iloc[:,0:]
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(amount.columns)))
                amount=pd.concat([amount,feat_df],axis=1)
                amount=amount[feat]
                amount=amount.fillna(0)
                
                amount.columns=pd.MultiIndex.from_product([["INGS"], amount.columns])
                
            if(dyn_csv.empty):
                dyn_csv= amount
            else:
                dyn_csv=pd.concat([dyn_csv,amount],axis=1)
            
            
        ###PROCS
        if(feat_proc):
            feat = final_proc['itemid'].unique()
            df2 = final_proc[final_proc['stay_id']==hid]
            
            if df2.shape[0]==0:
                hot=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                hot=hot.fillna(0)
                hot.columns=pd.MultiIndex.from_product([["PROC"], hot.columns])
            else:
                df2['val']=1
                hot=df2.pivot_table(index='start_time',columns='itemid',values='val')
                df2=df2.pivot_table(index='start_time',columns='itemid',values='stop_time')
    
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.ffill()
                df2=df2.fillna(0)

                hot=pd.concat([hot, add_df])
                hot=hot.sort_index()
                hot=hot.ffill()
                hot=hot.fillna(0)
            
                df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                df2[df2>=0]=1
                df2[df2<0]=0
                hot.iloc[:,0:]=df2.iloc[:,0:]*hot.iloc[:,0:]
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(hot.columns)))
                hot=pd.concat([hot,feat_df],axis=1)
                hot=hot[feat]
                hot=hot.fillna(0)
                
                hot.columns=pd.MultiIndex.from_product([["PROC"], hot.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=hot
                else:
                    dyn_csv=pd.concat([dyn_csv,hot],axis=1)
            
            
        ###OUT
        if(feat_out):
            feat=final_out['itemid'].unique()
            df2=final_out[final_out['stay_id']==hid]
        
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["OUT"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='itemid', values='value')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')

                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                val=val.fillna(0)

                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["OUT"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
                
            
        ###CHART
        if(feat_chart):
            feat=final_chart['itemid'].unique()
            df2=final_chart[final_chart['stay_id']==hid]
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                
                ## ECMO
                if 'ECMO' in val.columns:
                    val['ECMO'] = val['ECMO'].notna().astype(int)
                    
                ## Impella
                if 'Impella' in val.columns:
                    val['Impella'] = val['Impella'].notna().astype(int)
                    
                ## Catheter
                if 'Catheter' in val.columns:
                    val['Catheter'] = val['Catheter'].notna().astype(int)
                
                ## MAP
                val = val.ffill()
                val['MAP'] = (val['ABPs'] + 2*val['ABPd'])/3

                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
        
        ###LABS
        if(feat_lab):
            feat=final_labs['itemid'].unique()
            df2=final_labs[final_labs['stay_id']==hid]
            if df2.shape[0]==0:
                val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                val=val.fillna(0)
                val.columns=pd.MultiIndex.from_product([["LAB"], val.columns])
            else:
                val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)

                val=pd.concat([val, add_df])
                val=val.sort_index()
                
                val=val.ffill()

                df2[df2>0]=1
                df2[df2<0]=0
                
                feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                val=pd.concat([val,feat_df],axis=1)

                val=val[feat]
                val.columns=pd.MultiIndex.from_product([["LAB"], val.columns])
            
            if(dyn_csv.empty):
                dyn_csv=val
            else:
                dyn_csv=pd.concat([dyn_csv,val],axis=1)
                
                
        ###MICRO LABS       
        if(feat_micro):
            feat = final_micro['spec_itemid'].unique()
            df2 = final_micro[final_micro['stay_id']==hid]
            
            if df2.shape[0]==0:
                df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                df2=df2.fillna(0)
                df2.columns=pd.MultiIndex.from_product([["MICRO"], df2.columns])
            else:
                df2['val']=1
                #print(df2)
                df2=df2.pivot_table(index='start_time',columns='spec_itemid',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)
                df2[df2>0]=1

                feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                df2=pd.concat([df2,feat_df],axis=1)

                df2=df2[feat]
                df2=df2.fillna(0)
                df2.columns=pd.MultiIndex.from_product([["MICRO"], df2.columns])
            
            if(dyn_csv.empty):
                dyn_csv=df2
            else:
                dyn_csv=pd.concat([dyn_csv,df2],axis=1)
        
        #[ ====== Save temporal data to csv ====== ]
        
        dyn_csv.to_csv(local+'/csv/'+str(hid)+'/dynamic_proc.csv',index=False)
        
    print("[ SUCCESSFULLY SAVED TOTAL UNIT STAY DATA ]")