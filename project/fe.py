import pandas as pd    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import NearMiss

numeric_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
exclude_columns = ['encounter_id','patient_nbr']

def get_diag_mapping(diag1, diag2, diag3):
    dict={}
    dict['785'] = 'Circulatory'
    for i in range(390,460):
        dict[str(i)] = 'Circulatory'

    dict['786'] = 'Respiratory'
    for i in range(460,520):
        dict[str(i)] = 'Respiratory'

    dict['787'] = 'Digestive'
    for i in range(520,580):
        dict[str(i)] = 'Digestive'

    for i in range(800,1000):
        dict[str(i)] = 'Injury'

    for i in range(710,740):
        dict[str(i)] = 'Musculoskeletal'

    dict['788'] = 'Genitourinary'
    for i in range(580,630):
        dict[str(i)] = 'Genitourinary'

    for i in range(140,240):
        dict[str(i)] = 'Neoplasms'

    diabetes_code = []
    for x in diag1:
        if x.startswith('250'):
            if x not in diabetes_code:
                diabetes_code.append(x)
    for x in diag2:
        if x.startswith('250'):
            if x not in diabetes_code:
                diabetes_code.append(x)
    for x in diag3:
        if x.startswith('250'):
            if x not in diabetes_code:
                diabetes_code.append(x)
    for i in diabetes_code:
        dict[str(i)] = 'Diabetes'

    other = [] 
    for x in diag1:
        if (x.startswith('E') or x.startswith('V')):
            if x not in other:
                other.append(x)
    for x in diag2:
        if (x.startswith('E') or x.startswith('V')):
            if x not in other:
                other.append(x)
    for x in diag3:
        if (x.startswith('E') or x.startswith('V')):
            if x not in other:
                other.append(x)  
    other.append(str(780))
    other.append(str(781))
    other.append(str(784))
    other.append(str(782))
    other.append(str(789))
    for i in range(790,800):
        other.append(str(i))
    for i in range(240,280):
        other.append(str(i))
    for i in range(680,710):
        other.append(str(i))
    for i in range(280,290):
        other.append(str(i))
    for i in range(290,320):
        other.append(str(i))
    for i in range(320,360):
        other.append(str(i))
    for i in range(630,680):
        other.append(str(i))
    for i in range(360,390):
        other.append(str(i))
    for i in range(740,760):
        other.append(str(i))
    for i in range(1,140):
        other.append(str(i))
    for i in other:
        dict[i] = 'Other'
    return dict
    print(dict)

if __name__ == "__main__":
    data = pd.read_csv(r"dataset_diabetes/diabetic_data.csv", sep=",")
    data = data.replace('?', np.nan)

    # Task 1
    data_X = data.iloc[:, 0:49]
    data_y = data.iloc[:, 49:50]

    label_mapping = {"NO": 0, '<30': 1, '>30': 2}
    data_y = data_y.iloc[:, 0].map(label_mapping)
    
    exclude_columns = exclude_columns + ['weight', 'payer_code','medical_specialty']

    exclude_columns = exclude_columns + ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 
            'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone']
    exclude_columns= exclude_columns+['diag_2', 'diag_3']

    category_columns = data_X.columns.to_list()
    for c in exclude_columns:
        category_columns.remove(c)
    for c in numeric_columns:
        category_columns.remove(c)

    top_10 = ['UNK','InternalMedicine','Emergency/Trauma',\
          'Family/GeneralPractice', 'Cardiology','Surgery-General' ,\
          'Nephrology','Orthopedics',\
          'Orthopedics-Reconstructive','Radiologist']
    data_X['med_spec'] = data_X['medical_specialty'].copy()
    data_X.loc[~data_X.med_spec.isin(top_10),'med_spec'] = 'Other'
    category_columns.append('med_spec')

    data_X = data_X.drop(exclude_columns, axis=1)
   
    diag_mapping =  get_diag_mapping(data['diag_1'].astype(str),data['diag_2'].astype(str),data['diag_3'].astype(str))
    data_X['diag_1'] = data_X['diag_1'].map(diag_mapping)
    # data_X['diag_2'] = data_X['diag_2'].map(diag_mapping)
    # data_X['diag_3'] = data_X['diag_3'].map(diag_mapping)

    data_X['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge' 
                                              for val in data_X['discharge_disposition_id']], index=data_X.index)
    data_X['admission_source_id'] = pd.Series(['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Other source' 
                                              for val in data_X['admission_source_id']], index=data_X.index)
    data_X['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type' 
                                              for val in data_X['admission_type_id']], index=data_X.index)

    for f_pair in [['A1Cresult', 'change'],['age', 'gender'],['age','med_spec'],['race','discharge_disposition_id']]:
        data_X['_'.join(f_pair)] = data_X[f_pair[0]]+data_X[f_pair[1]]
        category_columns.append('_'.join(f_pair))
    
    age_mapping = {"[0-10)": 1,"[10-20)": 1,"[20-30)":1,"[30-40)": 2,"[40-50)": 2,"[50-60)": 2,"[60-70)": 3, '[70-80)': 3, '[80-90)': 3, '[90-100)': 3}
    data_X['age'] = data_X['age'].map(age_mapping)
    label_columns=['age']
    for cat in label_columns:
        data_X[cat] = LabelEncoder().fit_transform(data_X[cat].astype(str))
    
    dummy_columns = category_columns
    for c in label_columns:
        dummy_columns.remove(c)

    data_X = pd.get_dummies(data_X, columns=dummy_columns)

    # data_X['number_outpatient'] = data_X['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
    # data_X['number_emergency'] = data_X['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
    # data_X['number_inpatient'] = data_X['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))

    data_X[numeric_columns] = StandardScaler().fit_transform(data_X[numeric_columns])

    print(data_y.value_counts())
    print(data_X.shape)

    model_smote = SMOTE(random_state=42)
    data_X,data_y = model_smote.fit_sample(data_X,data_y)

    print(data_X.shape)

    lr = LogisticRegression(penalty="l1",solver='liblinear').fit(data_X, data_y)
    model = SelectFromModel(lr, prefit=True)
    data_X = pd.DataFrame(model.transform(data_X))

    print(data_X.shape)

    data_fe = pd.concat([data_y, data_X], axis=1)

    print(data_y.value_counts())
    data_fe.to_csv(r'data_fe_task1.csv', mode='w+', index=False)


    # Task 2
    
    # data_X1 = data.iloc[:, 0:3]
    # data_X2 = data.iloc[:, 5:50]
    # data_X = pd.concat([data_X1, data_X2], axis=1)
    # data_y = data.iloc[:, 4:5]

    # label_mapping = {"[0-10)": 0,"[10-20)": 0,"[20-30)": 0,"[30-40)": 0,"[40-50)": 0,"[50-60)": 1,"[60-70)": 1, '[70-80)': 2, '[80-90)': 2, '[90-100)': 2}
    # data_y = data_y.iloc[:, 0].map(label_mapping)

    # exclude_columns = exclude_columns + ['weight', 'payer_code','medical_specialty']

    # exclude_columns = exclude_columns + ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
    #         'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
    #         'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 
    #         'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    #         'metformin-rosiglitazone', 'metformin-pioglitazone']
    # exclude_columns= exclude_columns+['diag_2', 'diag_3']

    # category_columns = data_X.columns.to_list()
    # for c in exclude_columns:
    #     category_columns.remove(c)
    # for c in numeric_columns:
    #     category_columns.remove(c)

    # top_10 = ['UNK','InternalMedicine','Emergency/Trauma',\
    #       'Family/GeneralPractice', 'Cardiology','Surgery-General' ,\
    #       'Nephrology','Orthopedics',\
    #       'Orthopedics-Reconstructive','Radiologist']
    # data_X['med_spec'] = data_X['medical_specialty'].copy()
    # data_X.loc[~data_X.med_spec.isin(top_10),'med_spec'] = 'Other'
    # category_columns.append('med_spec')

    # data_X = data_X.drop(exclude_columns, axis=1)
   
    # diag_mapping =  get_diag_mapping(data['diag_1'].astype(str),data['diag_2'].astype(str),data['diag_3'].astype(str))
    # data_X['diag_1'] = data_X['diag_1'].map(diag_mapping)
    # # data_X['diag_2'] = data_X['diag_2'].map(diag_mapping)
    # # data_X['diag_3'] = data_X['diag_3'].map(diag_mapping)

  
    # data_X['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge' 
    #                                           for val in data_X['discharge_disposition_id']], index=data_X.index)
    # data_X['admission_source_id'] = pd.Series(['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Other source' 
    #                                           for val in data_X['admission_source_id']], index=data_X.index)
    # data_X['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type' 
    #                                           for val in data_X['admission_type_id']], index=data_X.index)

    # for f_pair in [['A1Cresult', 'change'],['race','discharge_disposition_id']]:
    #     data_X['_'.join(f_pair)] = data_X[f_pair[0]]+data_X[f_pair[1]]
    #     category_columns.append('_'.join(f_pair))
    
    # label_columns=[]
    # dummy_columns = category_columns
    # for c in label_columns:
    #     dummy_columns.remove(c)

    # data_X = pd.get_dummies(data_X, columns=dummy_columns)

    # # data_X['number_outpatient'] = data_X['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
    # # data_X['number_emergency'] = data_X['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
    # # data_X['number_inpatient'] = data_X['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))

    # data_X[numeric_columns] = StandardScaler().fit_transform(data_X[numeric_columns])

    # print(data_y.value_counts())
    # print(data_X.shape)

    # model_smote = SMOTE(random_state=42)
    # data_X,data_y = model_smote.fit_sample(data_X,data_y)

    # print(data_X.shape)

    # # lr = LogisticRegression(penalty="l1",solver='liblinear').fit(data_X, data_y)
    # # model = SelectFromModel(lr, prefit=True)
    # # data_X = pd.DataFrame(model.transform(data_X))

    # print(data_X.shape)

    # data_fe = pd.concat([data_y, data_X], axis=1)
    # print(data_y.value_counts())

    # data_fe.to_csv(r'data_fe_task2.csv', mode='w+', index=False)

    plt.show()


