import pandas as pd    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler

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
    print(data.shape)
    # numeric_data = data[numeric_columns]
    # print(data.isna().sum())
    # numeric_data.hist()
    # fig, axs = plt.subplots(3,3)
    # category_data = data.drop(numeric_columns+exclude_columns, axis=1)
    # for i, c in enumerate(category_data.columns):
    #     print('-------------------------------------------')
    #     # print(category_data[c].value_counts())
    #     val_count = category_data[c].value_counts()
    #     index = i % 9        
    #     if i > 8 and index == 0:
    #         fig, axs = plt.subplots(3,3)   
    #     if(val_count.size > 5):
    #         val_count = val_count[0:5]
    #     for vid, value in enumerate(val_count): 
    #         axs[index//3][index % 3].text(value, vid, str(value))
    #     val_count.plot(kind='barh', ax=axs[index//3][index % 3], title=c)
    
    data_X = data.iloc[:, 0:49]

    exclude_columns = exclude_columns + ['examide', 'citoglipton', 'weight', 'payer_code']
    category_columns = data_X.columns.to_list()
    for c in exclude_columns:
        category_columns.remove(c)
    for c in numeric_columns:
        category_columns.remove(c)
    
    data_X = data_X.drop(exclude_columns, axis=1)
   
    diag_mapping =  get_diag_mapping(data_X['diag_1'].astype(str),data_X['diag_2'].astype(str),data_X['diag_3'].astype(str))
    data_X['diag_1'] = data_X['diag_1'].map(diag_mapping)
    data_X['diag_2'] = data_X['diag_2'].map(diag_mapping)
    data_X['diag_3'] = data_X['diag_3'].map(diag_mapping)

   
    for f_pair in [['A1Cresult', 'change']]:
        data_X['_'.join(f_pair)] = data_X[f_pair[0]]+data_X[f_pair[1]]
        category_columns.append('_'.join(f_pair))
    # data_X = pd.get_dummies(data_X, columns=category_columns)
    for cat in category_columns:
        data_X[cat] = LabelEncoder().fit_transform(data_X[cat].astype(str))
    
    
    data_X[numeric_columns] = StandardScaler().fit_transform(data_X[numeric_columns])

    # data_X = data_X.drop(['diag_2', 'diag_3'], axis=1)

    data_y = data.iloc[:, 49:50]
    label_mapping = {"NO": 0, '<30': 1, '>30': 2}
    data_y = data_y.iloc[:, 0].map(label_mapping)
    model_smote = SMOTEENN(random_state=42)
    data_X,data_y = model_smote.fit_sample(data_X,data_y) 
    data_fe = pd.concat([data_y, data_X], axis=1)

    print(data_y.value_counts())

    data_fe.to_csv(r'data_fe.csv', mode='w+', index=False)

    # plt.show()


