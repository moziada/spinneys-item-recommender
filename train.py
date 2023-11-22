from recommendation_module import *
import pandas as pd
from tqdm import tqdm

DATA_PATH = Path("data/Loyalty-03-10-2023/transactions data")
model = Item2Item()
train_files = os.listdir(DATA_PATH)

for file in tqdm(train_files):
    train_df = pd.read_parquet(DATA_PATH / file)
    model.partial_fit(train_df, user_col="Receipt No_", item_col="Item No_")

model.estimate_scores()
model.save_model(save_dir="Loyalty-03-10-2023")