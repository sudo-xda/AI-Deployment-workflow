import pickle
import numpy as np
with open('model_and_norm.pkl', 'rb') as file:loaded_model, loaded_norm_func = pickle.load(file) 
x_in=[[0.22,2.70,3.28,0.98,9.9]]
#x_in=[[0.17, 1.60, 3.39, 0.48, 9.5]]
norm=loaded_norm_func(x_in)
x=norm.reshape(1,-1)
y=loaded_model.predict(x)

print(y)