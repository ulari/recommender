"""
This scripts contains 
data and functions useful for the other scripts
"""
import pickle


with open('cos_rick.pkl', 'rb') as file:
   DISTANCE_MODEL = pickle.load(file)

with open('nmf_rick.pkl', 'rb') as file:
   NMF_MODEL = pickle.load(file)

MOVIES = [
    "Men in Black",
    "Inglorious Bastards", 
    "My names is Nobody",
    "John Wick four",
    "Pocahontas", 
    "The Fast and Furious (8)", 
    "Tom and Jerry",
    "5"
]

nmf_model = ...

cos_sim_model = ...

if __name__ == '__main__':
   print(__name__)
   print(DISTANCE_MODEL)
   print(NMF_MODEL)