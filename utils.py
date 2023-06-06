"""
This scripts contains 
data and functions useful for the other scripts
"""
import pickle
import sklearn

with open('cos_rick.pkl', 'rb') as file:
   DISTANCE_MODEL = pickle.load(file)

with open('nmf_rick.pkl', 'rb') as file:
   NMF_MODEL = pickle.load(file)
