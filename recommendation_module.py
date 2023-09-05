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
        self.item2item_matrix = None
        self.item2idx = None
        self.idx2item = None

        if load_dir:
            self.load_item2item_matrix(load_dir)

    def calc_item2item_matrix(self, data: pd.DataFrame, receipt_col: str, item_col: str, save_dir: str):
        df = data.copy()
        df.drop_duplicates(subset=[receipt_col, item_col], inplace=True)
        unique_items = df[item_col].unique()
        self.idx2item = {str(i):str(code) for i, code in enumerate(unique_items)}
        self.item2idx = {str(code):str(i) for i, code in enumerate(unique_items)}

        receipt_items = df.groupby(receipt_col)[item_col].apply(set)
        # filter out receipts with only one item
        receipt_items = receipt_items[receipt_items.apply(lambda x: len(x) > 1)]

        item_combinations = receipt_items.apply(combinations, args=(2,))
        freq_matrix = pd.DataFrame(data=np.zeros((len(unique_items), len(unique_items))), index=unique_items, columns=unique_items, dtype=int)
        for row in tqdm(item_combinations):
            for idx in row:
                freq_matrix.loc[idx] += 1
        
        adjusted_freq_matrix = freq_matrix.copy()
        adjusted_freq_matrix.loc[:] += np.triu(adjusted_freq_matrix).T
        adjusted_freq_matrix.loc[:] = np.tril(adjusted_freq_matrix, k=-1) + np.tril(adjusted_freq_matrix, k=-1).T

        self.item2item_matrix = sparse.csc_matrix(adjusted_freq_matrix.values)
        if save_dir:
            self._save_item2item_matrix(save_dir)

    def _save_item2item_matrix(self, save_dir: str):
        full_path = Path("models") / save_dir
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        
        sparse.save_npz(full_path / "item2item-matrix.npz", self.item2item_matrix)
        with open(full_path / 'idx2item.pickle', 'wb') as handle:
            pickle.dump(self.idx2item, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_item2item_matrix(self, load_dir: str):
        full_path = Path("models") / load_dir
        self.item2item_matrix = sparse.load_npz(full_path / "item2item-matrix.npz")
        with open(full_path / 'idx2item.pickle', 'rb') as handle:
            self.idx2item = pickle.load(handle)
        self.item2idx = {v: k for k, v in self.idx2item.items()}

    def get_top_n_frequent_items(self, item_code: str, n=5) -> dict:
        idx = int(self.item2idx[item_code])
        item_scores = self.item2item_matrix[idx, :].toarray()
        top_n_idxs = np.argpartition(item_scores, -n)[0, -n:]
        ret_dict = {
            "items": [self.idx2item[str(i)] for i in top_n_idxs],
            "scores": item_scores[0, top_n_idxs]
            }
        return ret_dict


class Item2Subgroup():
    def __init__(self, load_dir: str = None):
        self.item2subgroup_matrix = None
        self.idx2item = None
        self.idx2subgroup = None
        self.item2subgroup_mapper = None
        
        if load_dir:
            self.load_item2subgroup(load_dir)

    def load_item2subgroup(self, load_dir):
        pass

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