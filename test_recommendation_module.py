import pytest
import numpy as np
import pandas as pd

from recommendation_module import Item2Item

def test_calc_item2item_matrix():
    model = Item2Item()
    df = pd.DataFrame({
        "Receipt No_": ["100", "100", "100", "100", "101", "101", "102"],
        "Item No_": ["10", "11", "12", "13", "10", "11", "10"]})
    model.calc_item2item_matrix(df, user_col="Receipt No_", item_col="Item No_")

    result = np.array([[0, 2/3, 1/3, 1/3], [2/3, 0, 0.5, 0.5], [1/3, 0.5, 0, 1], [1/3, 0.5, 1, 0]])
    assert np.allclose(result, model.item2item_matrix.toarray())


def test_get_top_n_frequent_items():
    model = Item2Item(load_dir="MOA-Jul-optimized-freq_adjusted-V01")
    expected_result = {
        'items': ['102248', '101695', '137346', '137220', '102251'],
        'scores': np.array([0.05806011, 0.04654545, 0.04359926, 0.04179663, 0.0415601])
        }
    
    result = model.get_top_n_frequent_items("100757", n=5)

    assert result['items'] == expected_result['items']
    assert np.allclose(result['scores'], expected_result['scores'])