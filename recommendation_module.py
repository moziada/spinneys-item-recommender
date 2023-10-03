import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
import pickle
from itertools import combinations, permutations
from tqdm import tqdm
import os

class Item2Item:
    def __init__(self, load_dir: str = None):
        self.item2item_frequency = pd.DataFrame()
        self.item2item_scores: sparse.csc_matrix = None
        
        self.item2idx:dict = None
        self.idx2item:dict = None

        if load_dir:
            self.load_item2item_matrix(load_dir)
    
    def partial_fit(self, data: pd.DataFrame, user_col: str, item_col: str):
        # filter out transactions with num items = 1
        df = pd.merge(data, data[[user_col]].value_counts().apply(lambda x: x > 1).rename("filter flag"), how="left", on=user_col)
        df = df[df["filter flag"]]

        # select columns to work on
        df = data.loc[:, [user_col, item_col]].drop_duplicates(subset=[user_col, item_col])

        interaction_matrix = df.merge(df, on=user_col, how="outer").groupby([item_col + "_x", item_col + "_y"]).size().unstack().fillna(0)
        
        self.item2item_frequency = self.item2item_frequency.add(interaction_matrix, fill_value=0).fillna(0)
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
        full_path = Path("models") / "item2item" / save_dir
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        
        sparse.save_npz(full_path / "item2item-scores.npz", self.item2item_scores)
        
        with open(full_path / 'idx2item.pickle', 'wb') as handle:
            pickle.dump(self.idx2item, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_item2item_matrix(self, load_dir: str):
        full_path = Path("models") / "item2item" / load_dir
        
        self.item2item_scores = sparse.load_npz(full_path / "item2item-scores.npz")
        
        with open(full_path / 'idx2item.pickle', 'rb') as handle:
            self.idx2item = pickle.load(handle)
        self.item2idx = {v: k for k, v in self.idx2item.items()}

    def get_top_n_frequent_items(self, item_code: str, n=5) -> dict:
        idx = self.item2idx.get(item_code)
        if not idx:
            return {"items": [], "scores": []}
        
        item_scores = np.squeeze(self.item2item_scores[idx, :].toarray())
        # Getting top n scores
        top_n_idxs = np.argpartition(item_scores, -n)[-n:]
        # Sorting top_n_idxs in descending order
        top_n_idxs = top_n_idxs[np.argsort(item_scores[top_n_idxs])[::-1]]
        ret_dict = {
            "items": [self.idx2item[i] for i in top_n_idxs],
            "scores": item_scores[top_n_idxs]
            }
        return ret_dict


class Item2Subgroup():
    def __init__(self, load_dir: str = None):
        self.item2subgroup_matrix = None
        
        self.idx2item = None
        self.item2idx = None
        
        self.idx2subgroup = None
        self.subgroup2idx = None

        self.item2subgroup_mapper = None
        
        if load_dir:
            self.load_item2subgroup(load_dir)

    def load_item2subgroup(self, load_dir):
        full_path = Path("models") / "item2subgroup" / load_dir
        self.item2subgroup_matrix = sparse.load_npz(full_path / "item2subgroup-matrix.npz")
        
        with open(full_path / 'idx2item.pickle', 'rb') as handle:
            self.idx2item = pickle.load(handle)
        self.item2idx = {v: k for k, v in self.idx2item.items()}

        with open(full_path / 'idx2subgroup.pickle', 'rb') as handle:
            self.idx2subgroup = pickle.load(handle)
        self.subgroup2idx = {v: k for k, v in self.idx2subgroup.items()}

        with open(full_path / 'item2subgroup_mapper.pickle', 'rb') as handle:
            self.item2subgroup_mapper = pickle.load(handle)

    def calc_item2subgroup_matrix(self, data: pd.DataFrame, receipt_col: str, item_col: str, subgroup_col: str, save_dir: str):
        df = data.copy()
        df.drop_duplicates(subset=[receipt_col, item_col], inplace=True)
        
        unique_items = df[item_col].unique()
        self.idx2item = {str(i):str(code) for i, code in enumerate(unique_items)}
        
        unique_subgroups = df[subgroup_col].unique()
        self.idx2subgroup = {str(i):str(code) for i, code in enumerate(unique_subgroups)}
        
        item2subgroup_mapper = df[["Item No_", "Item Sub Group Code"]].drop_duplicates(subset=["Item No_"]).set_index("Item No_").to_dict()["Item Sub Group Code"]

        receipt_items = df.groupby(receipt_col)[item_col].apply(set)
        # filter out receipts with only one item
        receipt_items = receipt_items[receipt_items.apply(lambda x: len(x) > 1)]

        item_permutations = receipt_items.apply(permutations, args=(2,))
        freq_matrix = pd.DataFrame(data=np.zeros((len(unique_items), len(unique_subgroups))), index=unique_items, columns=unique_subgroups, dtype=int)
        for row in tqdm(item_permutations):
            for i1, i2 in row:
                freq_matrix.loc[i1, item2subgroup_mapper[i2]] += 1

        self.item2subgroup_matrix = sparse.csc_matrix(freq_matrix.values)
        if save_dir:
            self._save_item2subgroup_matrix(save_dir)

    def _save_item2subgroup_matrix(self, save_dir: str):
        full_path = Path("models") / "item2subgroup" / save_dir
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        
        sparse.save_npz(full_path / "item2subgroup-matrix.npz", self.item2subgroup_matrix)
        
        with open(full_path / 'idx2item.pickle', 'wb') as handle:
            pickle.dump(self.idx2item, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(full_path / 'idx2subgroup.pickle', 'wb') as handle:
            pickle.dump(self.idx2subgroup, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(full_path / "item2subgroup_mapper.pickle", 'wb') as handle:
            pickle.dump(self.item2subgroup_mapper, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_top_n_frequent_subgroups(self, item_code: str, n=5) -> dict:
        idx = self.item2idx[item_code]
        subgroup_scores = self.item2subgroup_matrix[idx, :].toarray()
        top_n_idxs = np.argpartition(subgroup_scores, -n)[0, -n:]
        ret_dict = {
            "subgroups": [self.idx2subgroup[str(i)] for i in top_n_idxs],
            "scores": subgroup_scores[0, top_n_idxs]
            }
        return ret_dict