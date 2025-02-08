import pandas as pd
import numpy as np
import pickle
import json
import config


class predict_result():

    def __init__(self, user_data):
        self.model_file_path = 'adaboost_clf_model_file.pkl'
        self.user_data = user_data

    def load_saved_data(self):

        with open (self.model_file_path, 'rb') as f:
            self.model = pickle.load(f)

        with open ('project_data.json', 'r') as f:
            self.proj_data = json.load(f)

    def prediction(self):

        self.load_saved_data()

        gender         = self.user_data['gender']
        ever_married   = self.user_data['ever_married']
        Residence_type = self.user_data['Residence_type']

        gender         = self.proj_data['gender'][gender]
        ever_married   = self.proj_data['ever_married'][ever_married]
        Residence_type = self.proj_data['Residence_type'][Residence_type]

        work_type = 'work_type_' + self.user_data['work_type']
        work_type_index = np.where(self.proj_data['Columns'] == work_type)[0]
        

        smoking_status = 'smoking_status_' + self.user_data['smoking_status']
        smoking_status_index = np.where(self.proj_data['Columns'] == smoking_status)[0]

        col_len = len(self.proj_data['Column'])
        print(col_len)

        test_array = np.zeros(col_len)
        print(test_array)

        test_array[0] = gender
        test_array[1] = eval(self.user_data['age'])
        test_array[2] = eval(self.user_data['hypertension'])
        test_array[3] = eval(self.user_data['heart_disease'])
        test_array[4] = ever_married
        test_array[5] = Residence_type
        test_array[6] = eval(self.user_data['avg_glucose_level'])
        test_array[7] = eval(self.user_data['bmi'])
        test_array[work_type_index] = 1
        test_array[smoking_status_index] = 1

        print(test_array)

        predict_stroke = self.model.predict([test_array])[0]
        print('predict_stroke :', predict_stroke)
        return predict_stroke
    

if __name__  == '__main__':
    obj = predict_result()
    obj