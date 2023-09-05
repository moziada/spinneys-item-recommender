import pytest
import numpy as np

from recommendation_module import Item2Item

def test_get_top_n_frequent_items():
    model = Item2Item(load_dir="MOA - Jul")  # Create an instance of your class
    expected_result = {
        'items': ['111195', '388101', '102248', '287238', '112212'],
        'scores': np.array([70, 71, 85, 82, 86])
        }
    result = model.get_top_n_frequent_items("100757", n=5)
    assert result['items'] == expected_result['items']
    assert np.array_equal(result['scores'], expected_result['scores'])