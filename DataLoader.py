import wfdb
import numpy as np
import pandas as pd
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic, agg_df):
   
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def prepare_data(base_path, sampling_rate=100):
    
    # Load and convert annotation data
    Y = pd.read_csv(base_path + r'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, base_path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(base_path + r'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['class'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

    # Add additional diagnostic class
    Y['class2'] = Y['scp_codes'].apply(lambda x: max(x, key=x.get))

    return X, Y

def loader(base_path, sampling_rate=100):
    base_path = base_path
    sampling_rate = sampling_rate

    ecg_data, patient_data = prepare_data(base_path, sampling_rate)

    print("ECG Data Shape:", ecg_data.shape)
    print("Patient Data Head:")
    print(patient_data.head())
    return ecg_data, patient_data

#######################################

## This next bit maps the classes

def map_classes(patient_data):
   
    class_mapping = {
        'NORM': 'Normal ECG',
        'MI': 'Myocardial Infarction',
        'STTC': 'ST/T Change',
        'CD': 'Conduction Disturbance',
        'HYP': 'Hypertrophy'
    }

    # Flatten the list of classes and map to the first matching class
    patient_data['mapped_class'] = patient_data['class'].apply(
        lambda classes: next((class_mapping[c] for c in classes if c in class_mapping), 'Other')
    )
    return patient_data