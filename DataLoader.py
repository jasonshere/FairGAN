import os
import pandas as pd
import numpy as np
import tensorflow as tf


class AmazonDatasetsLoader(object):

    DATASETS = {
        'Amazon-toys-and-games': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Toys_and_Games.csv',
        'Amazon-beauty': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv',
        'Amazon-office-products': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Office_Products.csv',
        'Amazon-digital-music': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Digital_Music.csv',
    }

    def ratings(self, ds_name, UN=20, IN=10):
        dataset_file = tf.keras.utils.get_file(fname=os.path.basename(self.DATASETS[ds_name]),
                                       origin=self.DATASETS[ds_name],
                                       extract=False)
        
        rts = pd.read_csv(dataset_file, sep=',', names=['user_id', 'item_id', 'rating', 'timestamp'])
        rts = rts.drop_duplicates(subset=['item_id', 'user_id'], keep='last')
        while True:
            rts = rts[rts.groupby('user_id')['user_id'].transform('count') >= UN]
            rts = rts[rts.groupby('item_id')['item_id'].transform('count') >= IN]
            print(rts.groupby('user_id').count().min().rating, rts.groupby('item_id').count().min().rating)
            if (rts.groupby('user_id').count().min().rating >= UN) and rts.groupby('item_id').count().min().rating >= IN:
                break
            if np.isnan(rts.groupby('user_id').count().min().rating) or np.isnan(rts.groupby('item_id').count().min().rating):
                break
        # return rts[rts.user_id.isin(uids) & rts.item_id.isin(iids)]
        return rts