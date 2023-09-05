import pandas as pd
import os

#df = pd.read_csv("data/MOA-JUL V02.csv")
# print(df.drop_duplicates(subset=["Item No_", "Item Sub Group Code"]).head().to_dict(orient="list"))
#print(df[["Item No_", "Item Sub Group Code"]].drop_duplicates(subset=["Item No_"]).set_index("Item No_").head().to_dict())
os.makedirs("models\\item2subgroup\\MOA - Jul")
