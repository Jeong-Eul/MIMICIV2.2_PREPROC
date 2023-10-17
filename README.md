# MIMIC-IV preprocessing  

Source: Gupta, Mehak, et al. "An extensive data processing pipeline for mimic-iv." Machine Learning for Health. PMLR, 2022.    
Github: <a herf='https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main'>https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main</a>  


## [Code developing]  
MIMIC-IV 전처리에 관한 논문을 읽고, 이를 MIMIC-IV 2.2 Version 에 호환할 수 있도록 수정하였습니다.
 - MIMIC IV 2.0에는 존재하지 않는 ingredientevent.csv 를 반영할 수 있도록 feature_selection_icu.py를 대폭 수정하였습니다.  
 - ICU EHR에서도 Lab value를 활용할 수 있도록 수정하였습니다.    
 - Labevent를 반영하기 위해 stay_id를 부여하는 로직을 추가하였습니다.  
 - 파생 변수 제작이 용이하도록 각 Stay가 폴더 단위로 저장되는 기존의 방식을 하나의 데이터프레임으로 생성할 수 있도록 수정하였습니다.  
 - 상기 내용을 모두 구현하고, Circulatory Failure에 대한 조기 예측을 위해 mainCircline.ipynb 에서 전처리 과정을 추가하였습니다.  