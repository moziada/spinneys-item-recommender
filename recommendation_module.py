import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import pickle
import os
from recommendation_result import RecommendationResult

class Item2Item:
    def __init__(self, model_name: str = None):
        self.item2item_frequency = pd.DataFrame()
        self.item2item_scores: sparse.csc_matrix = None
        
        self.item2idx:dict = None
        self.idx2item:dict = None

        if model_name:
            self.load_item2item_matrix(model_name)
            self.items_info_dir = Path("data") / model_name / "items data" / "ItemInfo.parquet"
    
    def partial_fit(self, data: pd.DataFrame, user_col: str, item_col: str):
        # filter out transactions with num items = 1
        df = pd.merge(data, data[[user_col]].value_counts().apply(lambda x: x > 1).rename("filter flag"), how="left", on=user_col)
        df = df[df["filter flag"]]

        # select columns to work on
        df = data.loc[:, [user_col, item_col]].drop_duplicates(subset=[user_col, item_col])

        interaction_matrix = df.merge(df, on=user_col, how="outer").groupby([item_col + "_x", item_col + "_y"], observed=True).size().unstack().fillna(0).astype("uint32")
        
        self.item2item_frequency = self.item2item_frequency.add(interaction_matrix, fill_value=0).fillna(0).astype("uint32")
        assert (self.item2item_frequency.index == self.item2item_frequency.columns).all()

    def estimate_scores(self):
        self.idx2item = {i:str(code) for i, code in enumerate(self.item2item_frequency.index)}
        self.item2idx = {str(code):i for i, code in enumerate(self.item2item_frequency.index)}

        item_frequency = np.diagonal(self.item2item_frequency.values)
        # Subtracting intersection of items is to make score ranges from 0.0 to 1.0
        item2item_frequency_union = item_frequency + item_frequency[np.newaxis].T - self.item2item_frequency.values
        
        intersection = self.item2item_frequency.values
        np.fill_diagonal(intersection, 0)
        self.item2item_scores = sparse.csc_matrix(intersection / item2item_frequency_union)

    def fit(self, data: pd.DataFrame, user_col: str, item_col: str):
        self.partial_fit(data, user_col, item_col)
        self.estimate_scores()
    
    def save_model(self, save_dir: str):
        model_path = Path("models") / "item2item" / save_dir
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        sparse.save_npz(model_path / "item2item-scores.npz", self.item2item_scores)
        
        with open(model_path / 'idx2item.pickle', 'wb') as handle:
            pickle.dump(self.idx2item, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_item2item_matrix(self, load_dir: str):
        full_path = Path("models") / "item2item" / load_dir
        
        self.item2item_scores = sparse.load_npz(full_path / "item2item-scores.npz")
        
        with open(full_path / 'idx2item.pickle', 'rb') as handle:
            self.idx2item = pickle.load(handle)
        self.item2idx = {v: k for k, v in self.idx2item.items()}

    def get_top_n_frequent_items(self, item_code: str, n=5) -> RecommendationResult:
        idx = self.item2idx.get(item_code)
        if not idx:
            return RecommendationResult([], [], self.items_info_dir)
        
        item_scores = np.squeeze(self.item2item_scores[idx, :].toarray())
        # Getting top n scores
        top_n_idxs = np.argpartition(item_scores, -n)[-n:]
        # Sorting top_n_idxs in descending order
        top_n_idxs = top_n_idxs[np.argsort(item_scores[top_n_idxs])[::-1]]
        ret_dict = {
            "items": np.array([self.idx2item[i] for i in top_n_idxs]),
            "scores": item_scores[top_n_idxs]
            }
        return RecommendationResult(ret_dict["items"], ret_dict["scores"], self.items_info_dir)
    