import os
import pickle
import glob
import importlib
#print(os.getcwd())
#os.chdir('../../')
#print(os.getcwd())
import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *

local = '/Users/DAHS/Desktop/MIMICIV2.2_PREPROC/data'

if not os.path.exists(local+"/features"):
    os.makedirs(local+"/features")
  
def feature_icu(root_dir, cohort_output, version_path, diag_flag=True,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True, ing_flag=True, lab_flag=True, microlab_flag=True):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module(root_dir+version_path+"/hosp/diagnoses_icd.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
        diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code','root_icd10_convert','root']].to_csv(local+"/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if out_flag:  
        print("[EXTRACTING OUPTPUT EVENTS DATA]")
        out = preproc_out(root_dir+version_path+"/icu/outputevents.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
        out = drop_wrong_uom(out, 0.95)
        out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit', 'value']].to_csv(local+"/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    
    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart=preproc_chart(root_dir+version_path+"/icu/chartevents.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
        chart = drop_wrong_uom(chart, 0.95)
        chart[['stay_id', 'itemid','charttime','event_time_from_admit','valuenum']].to_csv(local+"/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
        
    if lab_flag:
        print("[EXTRACTING LABS DATA]")
        lab = preproc_labs(root_dir+version_path+"/hosp/labevents.csv.gz", version_path, local+'/cohort/'+cohort_output+'.csv.gz','charttime', 'base_anchor_year', dtypes=None, usecols=None)
        lab = drop_wrong_uom(lab, 0.95)
        lab[['subject_id', 'hadm_id','charttime','itemid','lab_time_from_admit','valuenum']].to_csv(local+'/features/preproc_labs.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")
        
    if microlab_flag:
        print("[EXTRACTING MICRO LABS DATA]")
        lab = preproc_microlabs(root_dir+version_path+"/hosp/microbiologyevents.csv.gz", version_path, local+'/cohort/'+cohort_output+'.csv.gz','charttime', 'base_anchor_year', dtypes=None, usecols=None)
        lab[['subject_id', 'hadm_id','charttime','spec_itemid','lab_time_from_admit']].to_csv(local+'/features/preproc_microlabs.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LABS DATA]")
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc(root_dir+version_path+"/icu/procedureevents.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz', 'starttime', dtypes=None, usecols=['stay_id','starttime','endtime','itemid'])
        proc[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit', 'stop_hours_from_admit']].to_csv(local+"/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds(root_dir+version_path+"/icu/inputevents.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        med[['subject_id', 'hadm_id', 'stay_id', 'itemid' ,'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate','amount','orderid']].to_csv(local+'/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
        
    if ing_flag:
        print("[EXTRACTING INGREDIENTS DATA]")
        ing = preproc_ings(root_dir+version_path+"/icu/ingredientevents.csv.gz", local+'/cohort/'+cohort_output+'.csv.gz')
        ing[['subject_id', 'hadm_id', 'stay_id', 'itemid' ,'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate','amount','orderid']].to_csv(local+'/features/preproc_ing_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED INGREDIENTS DATA]")
  



def preprocess_features_icu(cohort_output, diag_flag, group_diag,chart_flag,clean_chart,impute_outlier_chart,thresh,left_thresh,
                            lab_flag, imput_outlier_lab, thresh_lab, left_thresh_lab, clean_labs):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(local+"/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        if(group_diag=='Keep both ICD-9 and ICD-10 codes'):
            diag['new_icd_code']=diag['icd_code']
        if(group_diag=='Convert ICD-9 to ICD-10 codes'):
            diag['new_icd_code']=diag['root_icd10_convert']
        if(group_diag=='Convert ICD-9 to ICD-10 and group ICD-10 codes'):
            diag['new_icd_code']=diag['root']

        diag=diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
        print("Total number of rows",diag.shape[0])
        diag.to_csv(local+"/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        
    if chart_flag:
        if clean_chart:   
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv(local+"/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
            chart = outlier_imputation(chart, 'itemid', 'valuenum', thresh,left_thresh,impute_outlier_chart)
            
#             for i in [227441, 229357, 229358, 229360]:
#                 try:
#                     maj = chart.loc[chart.itemid == i].valueuom.value_counts().index[0]
#                     chart = chart.loc[~((chart.itemid == i) & (chart.valueuom == maj))]
#                 except IndexError:
#                     print(f"{idx} not found")
            print("Total number of rows",chart.shape[0])
            chart.to_csv(local+"/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
    if lab_flag:
        
        if clean_labs:   
            print("[PROCESSING LABS DATA]")
            labs = pd.read_csv(local+"/features/preproc_labs.csv.gz", compression='gzip',header=0)
            labs = outlier_imputation(labs, 'itemid', 'valuenum', thresh_lab,left_thresh_lab,imput_outlier_lab)
            

#             for i in [51249, 51282]:
#                 try:
#                     maj = labs.loc[labs.itemid == i].valueuom.value_counts().index[0]
#                     labs = labs.loc[~((labs.itemid == i) & (labs.valueuom == maj))]
#                 except IndexError:
#                     print(f"{idx} not found")
            print("Total number of rows",labs.shape[0])
#             del labs['valueuom']
            labs.to_csv(local+"/features/preproc_labs.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")        
    
def generate_summary_icu(local_dir , diag_flag,proc_flag,med_flag,out_flag,chart_flag,ing_flag,lab_flag, micro_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv(local_dir+'/'+"features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        freq=diag.groupby(['stay_id','new_icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
        total=diag.groupby('new_icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='new_icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/diag_summary.csv',index=False)
        summary['new_icd_code'].to_csv(local_dir+'/'+'summary/diag_features.csv',index=False)


    if med_flag:
        med = pd.read_csv(local_dir+'/'+"features/preproc_med_icu.csv.gz", compression='gzip',header=0)
        freq=med.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=med[med['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=med.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/med_summary.csv',index=False)
        summary['itemid'].to_csv(local_dir+'/'+'summary/med_features.csv',index=False)
        
    if ing_flag:
        ing = pd.read_csv(local_dir+'/'+"features/preproc_ing_icu.csv.gz", compression='gzip',header=0)
        freq=ing.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=ing[ing['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=ing.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/ing_summary.csv',index=False)
        summary['itemid'].to_csv(local_dir+'/'+'summary/ing_features.csv',index=False)
    
    if proc_flag:
        proc = pd.read_csv(local_dir+'/'+"features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
        freq=proc.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=proc.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/proc_summary.csv',index=False)
        summary['itemid'].to_csv(local_dir+'/'+'summary/proc_features.csv',index=False)

        
    if out_flag:
        out = pd.read_csv(local_dir+'/'+"features/preproc_out_icu.csv.gz", compression='gzip',header=0)
        freq=out.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=out.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/out_summary.csv',index=False)
        summary['itemid'].to_csv(local_dir+'/'+'summary/out_features.csv',index=False)
        
    if chart_flag:
        chart=pd.read_csv(local_dir+'/'+"features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
        freq=chart.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

        missing=chart[chart['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=chart.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        #summary=summary.fillna(0)

#         final.groupby('itemid')['missing_count'].sum().reset_index()
#         final.groupby('itemid')['total_count'].sum().reset_index()
#         final.groupby('itemid')['missing%'].mean().reset_index()
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/chart_summary.csv',index=False)
        summary['itemid'].to_csv(local_dir+'/'+'summary/chart_features.csv',index=False)
    
    if lab_flag:
        chunksize = 10000000
        labs=pd.DataFrame()
        for chunk in tqdm(pd.read_csv(local_dir+'/'+"features/preproc_labs.csv.gz", compression='gzip',header=0, index_col=None,chunksize=chunksize)):
            if labs.empty:
                labs=chunk
            else:
                labs=labs.append(chunk, ignore_index=True)
        freq=labs.groupby(['hadm_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=labs[labs['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=labs.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/labs_summary.csv',index=False)
        summary['itemid'].to_csv(local_dir+'/'+'summary/labs_features.csv',index=False)
        
    if micro_flag:
        microlabs = pd.read_csv(local_dir+'/'+"features/preproc_microlabs.csv.gz", compression='gzip',header=0)
        freq=microlabs.groupby(['stay_id','spec_itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['spec_itemid'])['mean_frequency'].mean().reset_index()
        total=microlabs.groupby('spec_itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='spec_itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv(local_dir+'/'+'summary/microlabs_summary.csv',index=False)
        summary['spec_itemid'].to_csv(local_dir+'/'+'summary/microlabs_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")  
    
    
    
def features_selection_icu(local_dir, cohort_output, diag_flag, proc_flag, med_flag, ing_flag, out_flag, lab_flag, chart_flag, micro_flag,
                           group_diag, group_med, group_ing, group_proc, group_out, group_chart, clean_labs):
    
    features=pd.read_csv(local_dir+"/summary/total_item_id.csv",header=0)
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv(local_dir+"/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
            diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
            print("Total number of rows",diag.shape[0])
            diag.to_csv(local_dir+"/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv(local_dir+"/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            med=med[med['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            rename_dict = dict(zip(features.itemid, features.rename_n))
            med['itemid'] = med['itemid'].map(rename_dict)
            med.to_csv(local_dir+'/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
            
    if ing_flag:       
        if group_ing:   
            print("[FEATURE SELECTION INGREDIENTS DATA]")
            ing = pd.read_csv(local_dir+"/features/preproc_ing_icu.csv.gz", compression='gzip',header=0)
            ing=ing[ing['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            rename_dict = dict(zip(features.itemid, features.rename_n))
            ing['itemid'] = ing['itemid'].map(rename_dict)
            ing.to_csv(local_dir+'/features/preproc_ing_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED INGREDIENTS DATA]")
    
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv(local_dir+"/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            proc=proc[proc['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",proc.shape[0])
            rename_dict = dict(zip(features.itemid, features.rename_n))
            proc['itemid'] = proc['itemid'].map(rename_dict)
            proc.to_csv(local_dir+"/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv(local_dir+"/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            out=out[out['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",out.shape[0])
            rename_dict = dict(zip(features.itemid, features.rename_n))
            out['itemid'] = out['itemid'].map(rename_dict)
            out.to_csv(local_dir+"/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv(local_dir+"/features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            chart=chart[chart['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",chart.shape[0])
            rename_dict = dict(zip(features.itemid, features.rename_n))
            chart['itemid'] = chart['itemid'].map(rename_dict)
            chart.to_csv(local_dir+"/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
    if lab_flag:
        if clean_labs:            
            print("[FEATURE SELECTION LABS DATA]")
            chunksize = 10000000
            labs=pd.DataFrame()
            for chunk in tqdm(pd.read_csv(local_dir+"/features/preproc_labs.csv.gz", compression='gzip',header=0, index_col=None,chunksize=chunksize)):
                if labs.empty:
                    labs=chunk
                else:
                    labs=labs.append(chunk, ignore_index=True)
            labs=labs[labs['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",labs.shape[0])
            rename_dict = dict(zip(features.itemid, features.rename_n))
            labs['itemid'] = labs['itemid'].map(rename_dict)
            labs.to_csv(local_dir+"/features/preproc_labs.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED LABS DATA]")
            
            
    if micro_flag:
        print("[FEATURE SELECTION MICRO LABS DATA]")
        chunksize = 500000
        labs=pd.DataFrame()
        for chunk in tqdm(pd.read_csv(local_dir+"/features/preproc_microlabs.csv.gz", compression='gzip',header=0, index_col=None,chunksize=chunksize)):
            if labs.empty:
                labs=chunk
            else:
                labs=labs.append(chunk, ignore_index=True)
        labs=labs[labs['spec_itemid'].isin(features['itemid'].unique())]
        print("Total number of rows",labs.shape[0])
        rename_dict = dict(zip(features.itemid, features.rename_n))
        labs['itemid'] = labs['itemid'].map(rename_dict)
        labs.to_csv(local_dir+"/features/preproc_microlabs.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MICRO LABS DATA]")