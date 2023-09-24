import pytest
import numpy as np
import pandas as pd

from recommendation_module import Item2Item

def test_fit():
    model = Item2Item()
    df = pd.DataFrame({
        "Receipt No_": ["100", "100", "100", "100", "101", "101", "102"],
        "Item No_": ["10", "11", "12", "13", "10", "11", "10"]})
    model.fit(df, user_col="Receipt No_", item_col="Item No_")

    result = np.array([[0, 2/3, 1/3, 1/3], [2/3, 0, 0.5, 0.5], [1/3, 0.5, 0, 1], [1/3, 0.5, 1, 0]])
    assert np.allclose(result, model.lift.toarray())

def test_partial_fit():
    model = Item2Item()
    
    df1 = pd.DataFrame({
        "Receipt No_": ["100", "100"],
        "Item No_": ["a", "b"]})
    
    df2 = pd.DataFrame({
        "Receipt No_": ["101", "101"],
        "Item No_": ["a", "b"]})
    
    df3 = pd.DataFrame({
        "Receipt No_": ["102", "102"],
        "Item No_": ["a", "c"]})

    df4 = pd.DataFrame({
        "Receipt No_": ["103", "103"],
        "Item No_": ["c", "d"]})
    
    df5 = pd.DataFrame({
        "Receipt No_": ["104"],
        "Item No_": ["e"]})
    
    for df in [df1, df2, df3, df4, df5]:
        model.partial_fit(df, trans_col="Receipt No_", item_col="Item No_")
    
    model.estimate_scores()

    result = np.array([[0, 4/3, 2/3, 0], [4/3, 0, 0, 0], [2/3, 0, 0, 2], [0, 0, 2, 0]])
    print(model.lift)
    assert np.allclose(result, model.lift.toarray())

def test_get_top_n_frequent_items():
    model = Item2Item(load_dir="MOA-Jul-optimized-freq_adjusted-V01")
    expected_result = {
        'items': ['102248', '101695', '137346', '137220', '102251'],
        'scores': np.array([0.05806011, 0.04654545, 0.04359926, 0.04179663, 0.0415601])
        }
    
    result = model.get_top_n_frequent_items("100757", n=5)

    assert result['items'] == expected_result['items']
    assert np.allclose(result['scores'], expected_result['scores'])