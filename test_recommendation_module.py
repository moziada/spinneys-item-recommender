import pytest
import numpy as np
import pandas as pd
from operator import itemgetter

from recommendation_module import Item2Item

def test_calc_item2item_matrix():
    model = Item2Item()
    df = pd.DataFrame({
        "Receipt No_": ["100", "100", "100", "100", "101", "101", "102"],
        "Item No_": ["10", "11", "12", "13", "10", "11", "10"]})
    model.calc_item2item_matrix(df, user_col="Receipt No_", item_col="Item No_")

    result = np.array([[0, 0.4, 0.25, 0.25], [0.4, 0, 1/3, 1/3], [0.25, 1/3, 0, 0.5], [0.25, 1/3, 0.5, 0]])
    assert np.allclose(result, model.item2item_matrix.toarray())


def test_get_top_n_frequent_items():
    model = Item2Item(load_dir="MOA-Jul-optimized-V01")  # Create an instance of your class
    expected_result = {
        'items': ['111195', '388101', '287238', '102248', '112212'],
        'scores': np.array([70, 71, 82, 85, 86])
        }
    
    result = model.get_top_n_frequent_items("100757", n=5)
    order = np.argsort(result['scores'])

    assert result['items'] == list(itemgetter(*order)(expected_result['items']))
    assert np.array_equal(result['scores'], expected_result['scores'][order])