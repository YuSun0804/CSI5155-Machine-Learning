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
from scipy import stats

numeric_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
exclude_columns = ['encounter_id','patient_nbr']

if __name__ == "__main__":
    data = pd.read_csv(r"dataset_diabetes/diabetic_data.csv", sep=",")
    data = data.replace('?', np.nan)
    print(data.shape)
    print(data.isna().sum())

    data_X = data.iloc[:, 0:49]
    data_y = data.iloc[:, 49:50]

    numeric_data = data_X[numeric_columns]
    numeric_data.hist()
    print(numeric_data.skew())

    data_X['number_outpatient'] = data['number_outpatient'].apply(lambda x: np.log(x + 0.01))
    data_X['number_emergency'] = data['number_emergency'].apply(lambda x: np.log(x + 0.01))
    data_X['number_inpatient'] = data['number_inpatient'].apply(lambda x: np.log(x + 0.01))

    print(numeric_data.skew())

    plt.figure()
    numeric_data.boxplot()

    category_data = data_X.drop(numeric_columns+exclude_columns, axis=1)

    for i, c in enumerate(category_data.columns):
        val_count = category_data[c].value_counts()
        s = []
        s.append(c)
        s.append(str(len(val_count)))
        s.append(str(category_data[c].isna().sum()) +"("+str(np.around(category_data[c].isna().sum()/101766,2))+")")
        s.append(str(val_count.index[0]))
        s.append(str(val_count.iloc[0]) + "("+str(np.around(val_count.iloc[0]/101766,2))+")")
        print(" & ".join(s) + "\\\\")


    medication_colunmns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 
            'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone','insulin']
    medication_data = data_X[medication_colunmns]
    fig, axs = plt.subplots(3,3)
    for i, c in enumerate(medication_colunmns):
        # print(category_data[c].value_counts())
        val_count = medication_data[c].value_counts()
        index = i % 9
        if i > 8 and index == 0:
            fig, axs = plt.subplots(3,3)            
        if(val_count.size > 5):
            val_count = val_count[0:5]
        for vid, value in enumerate(val_count): 
            axs[index//3][index % 3].text(value, vid, str(value))
        val_count.plot(kind='barh', ax=axs[index//3][index % 3], title=c)

    # fig, axs = plt.subplots(9,3)
    # category_data = data_X.drop(numeric_columns+exclude_columns, axis=1)
    # for i, c in enumerate(category_data.columns):
    #     print('-------------------------------------------')
    #     # print(category_data[c].value_counts())
    #     val_count = category_data[c].value_counts()
    #     index = i % 27        
    #     if i > 26 and index == 0:
    #         fig, axs = plt.subplots(3,3)   
    #     if(val_count.size > 5):
    #         val_count = val_count[0:5]
    #     for vid, value in enumerate(val_count): 
    #         axs[index//3][index % 3].text(value, vid, str(value))
    #     val_count.plot(kind='barh', ax=axs[index//3][index % 3], title=c)
    
    plt.figure()
    data_y.value_counts().plot(kind='barh', title=data_y.columns[0])

    data_y = data.iloc[:, 4:5]
    plt.figure()
    data_y.value_counts().plot(kind='barh', title='original age')
    
    label_mapping = {"[0-10)": '[0-50)',"[10-20)": '[0-50)',"[20-30)": '[0-50)',"[30-40)": '[0-50)',
        "[40-50)": '[0-50)',"[50-60)": '[50-70)',"[60-70)": '[50-70)', '[70-80)': '[70-100)', '[80-90)': '[70-100)', '[90-100)': '[70-100)'}
    data_y = data_y.iloc[:, 0].map(label_mapping)
    plt.figure()
    data_y.value_counts().plot(kind='barh', title='grouped age')

    plt.show()


