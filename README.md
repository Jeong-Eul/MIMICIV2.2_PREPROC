# MIMIC-IV preprocessing  

Source: Gupta, Mehak, et al. "An extensive data processing pipeline for mimic-iv." Machine Learning for Health. PMLR, 2022.    
Github: <a herf='https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main'>https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main</a>  


## [Code developing]  
MIMIC-IV 전처리에 관한 논문을 읽고, 이를 MIMIC-IV 2.2 Version 에 호환할 수 있도록 수정하였습니다.
 - MIMIC IV 2.0에는 존재하지 않는 ingredientevent.csv 를 반영할 수 있도록 feature_selection_icu.py를 대폭 수정하였습니다.  
 - ICU EHR에서도 Lab value를 활용할 수 있도록 수정하였습니다.    
 - Labevent를 반영하기 위해 stay_id를 부여하는 로직을 추가하였습니다.  
 - 파생 변수 제작이 용이하도록 각 Stay가 폴더 단위로 저장되는 기존의 방식을 하나의 데이터프레임으로 생성할 수 있도록 수정하였습니다.  
 - 원래 구현 방식이 예측을 하기 위해 몇 개의 시퀀스의 데이터가 들어갈 것인지 미리 설정하고, 모든 stay가 같은 길이의 데이터로 출력이 되는데, 이러한 제약 조건을 해제하여 각 stay 별로 끝까지 데이터가 출력되게 수정하였습니다.  
 - 원래 구현 방식이 output event의 경우 값이 있으면 1 아니면 0으로 변환하도록 되어 있는데, 이를 해제하여 Urine output 같은 값의 양이 중요한 변수를 최대한 활용할 수 있도록 수정하였습니다.
 - 상기 내용을 모두 구현하고, Circulatory Failure에 대한 조기 예측을 위해 mainCircline.ipynb 에서 전처리 과정을 추가하였습니다.  