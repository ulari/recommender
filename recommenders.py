"""
In this script we define functions for the recommender web
application
"""
import random
from utils import DISTANCE_MODEL, NMF_MODEL
import pandas as pd
import numpy as np

ratings = pd.read_csv ('ratings.csv', index_col=0)

movies = ratings.columns.to_list()

def recommend_nmf(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
        
    # 1. construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=movies, index=['new_user'])
    # impute
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)

    # transform
    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    
    # Q matrix
    Q_matrix = model.components_

    # R hat
    R_hat_new_user_matrix = np.dot(P_new_user_matrix,Q_matrix)

    # Add labels etc
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=movies,
                         index = ['new_user'])

    # Remove movies already seen
    R_hat_new_user_filtered = R_hat_new_user.drop(query.keys(), axis=1)

    # Return top-k higherst rated movies
    ranked =  R_hat_new_user_filtered.T.sort_values(by='new_user', ascending=False).index.to_list()


    
    recommended = ranked[:k]
    return recommended


def recommend_neighborhood(query, model, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation
    new_user_dataframe =  pd.DataFrame(query, columns=movies, index=['new_user'])
    # impute
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)   
    # construct a user vector
    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = model.kneighbors(
        new_user_dataframe_imputed,
        n_neighbors=5,
        return_distance=True
    )

    # sklearn returns a list of predictions
    # extract the first and only value of the list

    neighbors_df = pd.DataFrame(
        data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )

    # Only similar neighbours
    neighborhood = ratings.iloc[neighbor_ids[0]]

    # Filter out seen movies
    neighborhood_filtered = neighborhood.drop(query.keys(),axis=1)

    # Avergae ratings of movies people have actually rated
    mask = neighborhood_filtered > 0
    df_score = neighborhood_filtered[mask].mean()

    # Highest ranked
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()

    recommendations = df_score_ranked[:k]
    
    return recommendations


