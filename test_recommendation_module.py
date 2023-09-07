import pytest
import numpy as np
from operator import itemgetter

from recommendation_module import Item2Item

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