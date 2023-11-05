# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:33:10 2023

@author: sadwika sabbella
"""

import pickle
model_sample = pickle.load(open(r'iris_model.pkl','rb'))
print(model_sample.predict([[1.2,2,2.6,3.2]]))