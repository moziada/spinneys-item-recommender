from recommendation_module import Item2Subgroup
import pandas as pd

df = pd.read_csv("data/MOA-JUL V02.csv")
model = Item2Subgroup()
model.calc_item2subgroup_matrix(df, save_dir="MOA - Jul", item_col="Item No_", receipt_col="Receipt No_", subgroup_col="Item Sub Group Code")