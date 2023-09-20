import pandas as pd

items_info_df = pd.read_excel("data/products info (shortlist) - 2.xlsx", dtype=str)

def post_ranking(item_code: str, model_out: dict, exclude_subgroup: bool, exclude_prodgroup: bool, item_categories: list, items_info_df=items_info_df):
    item_record = items_info_df[items_info_df["Item Code"]==item_code]
    items_info_df = items_info_df[items_info_df["Item Code"].isin(model_out["items"])]

    if exclude_subgroup:
        target_item_subgroup = item_record["Subgroup Code"].values[0]
        items_info_df = items_info_df[items_info_df["Subgroup Code"] != target_item_subgroup]
    
    if exclude_prodgroup:
        target_item_prodgroup = item_record["Group Code"].values[0]
        items_info_df = items_info_df[items_info_df["Group Code"] != target_item_prodgroup]
    
    if item_categories:
        items_info_df = items_info_df[items_info_df["Category Code"].isin(item_categories)]

    item_codes = []
    item_scores = []
    for i, _ in enumerate(model_out["items"]):
        if model_out["items"][i] in items_info_df["Item Code"].values:
            item_codes.append(model_out["items"][i])
            item_scores.append(model_out["scores"][i])
    
    return {"items": item_codes, "scores": item_scores}
