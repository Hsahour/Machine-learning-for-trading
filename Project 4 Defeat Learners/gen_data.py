""""""                                        
"""                                        
template for generating data to fool learners (c) 2016 Tucker Balch                                        
Copyright 2018, Georgia Institute of Technology (Georgia Tech)                                        
Atlanta, Georgia 30332                                        
All Rights Reserved                                        
                                        
Template code for CS 4646/7646                                        
                                        
Georgia Tech asserts copyright ownership of this template and all derivative                                        
works, including solutions to the projects assigned in this course. Students                                        
and other users of this template code are advised not to share it with others                                        
or to make it available on publicly viewable websites including repositories                                        
such as github and gitlab.  This copyright statement should not be removed                                        
or edited.                                        
                                        
We do grant permission to share solutions privately with non-students such                                        
as potential employers. However, sharing with other current or future                                        
students of CS 7646 is prohibited and subject to being investigated as a                                        
GT honor code violation.                                        
                                        
-----do not edit anything above this line---                                        
                                        
Student Name: Hossein Sahour                                         
GT User ID: hsahour3                                       
GT ID: 903941641                                        
"""                                        
                                        
import math                                                                             
import numpy as np                                        
                                        
                                                                                                                     
def best_4_lin_reg(seed=1489683273):
    np.random.seed(seed)
    rows = np.random.randint(10, 1001) 
    features = np.random.randint(2, 11)  
    x = np.random.rand(rows, features)  
    weights = np.random.rand(features)  
    y = np.dot(x, weights) + np.random.normal(0, 0.1, rows) 
    return x, y

def best_4_dt(seed=1489683273):
    np.random.seed(seed)
    rows = np.random.randint(10, 1001)
    features = np.random.randint(2, 11)
    x = np.random.rand(rows, features) * 100
    y = np.sin(x[:, 0]) * np.log1p(x[:, 1]) + np.random.normal(0, 5, rows)
    if features > 2:
        for i in range(2, features):
            y += np.power(x[:, i], i % 4 + 1) / (i + 1)
    return x, y

def author():                                                                  
    return "hsahour3" 
                                                                             
