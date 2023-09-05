from recommendation_module import Item2Item

model = Item2Item(load_dir="MOA - Jul")
print(model.get_top_n_frequent_items("125350", n=5))