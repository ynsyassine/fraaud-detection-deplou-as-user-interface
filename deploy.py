#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:45:59 2023

@author: yassine
"""

import numpy as np 
import pickle 
loaded_model  = pickle.load(open("/home/yassine/trained_model.sav" , "rb"))
feature = np.array([[2,9990.0 , 9999.0,9.0]])
print(loaded_model.predict(feature))