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

def rule_max_items_per_subgroup(model_out: dict, n: int=1):
    item_codes = []
    item_scores = []
    subgroup_count = {}
    for i, item_code in enumerate(model_out["items"]):
        subgroup_code = get_item_subgroup(item_code)
        if subgroup_count.get(subgroup_code, 0) < n:
            item_codes.append(model_out["items"][i])
            item_scores.append(model_out["scores"][i])
        subgroup_count[subgroup_code] = subgroup_count.get(subgroup_code, 0) + 1
    
    return {"items": item_codes, "scores": item_scores}

def rule_same_category(model_out: dict, antecedent_item_category):
    item_codes = []
    item_scores = []
    for i, item_code in enumerate(model_out["items"]):
         consequen_item_category = get_item_category(item_code)
         if antecedent_item_category == consequen_item_category:
            item_codes.append(model_out["items"][i])
            item_scores.append(model_out["scores"][i])

def rule_different_category(model_out: dict, antecedent_item_category):
    item_codes = []
    item_scores = []
    for i, item_code in enumerate(model_out["items"]):
         consequen_item_category = get_item_category(item_code)
         if antecedent_item_category != consequen_item_category:
            item_codes.append(model_out["items"][i])
            item_scores.append(model_out["scores"][i])

def rule_different_subgroup(model_out: dict, antecedent_item_subgroup):
    item_codes = []
    item_scores = []
    for i, item_code in enumerate(model_out["items"]):
         consequen_item_subgroup = get_item_subgroup(item_code)
         if antecedent_item_subgroup != consequen_item_subgroup:
            item_codes.append(model_out["items"][i])
            item_scores.append(model_out["scores"][i])

def apply_category_filters(model_out: dict, antecedent_item_code):
    antecedent_item_category = get_item_category(antecedent_item_code)
    # H&B category = 0104, GM division categories = 03xx
    if antecedent_item_category == "0104" or antecedent_item_category.startswith("03"):
        model_out = rule_same_category(model_out, antecedent_item_category)
    
    # if category is meat, recommend products out of meat category
    elif antecedent_item_category == "0209":
        model_out = rule_different_category(model_out, "0209")
    
    # if nothing from above, just remove recommendations from the same subgroup
    else:
        antecedent_item_subgroup = get_item_subgroup(antecedent_item_code)
        model_out = rule_different_subgroup(model_out, antecedent_item_subgroup)
    
    model_out = rule_max_items_per_subgroup(model_out, n=1)
    return model_out
    
def get_item_category(item_code: str) -> str:
    item_record = items_info_df[items_info_df["Item Code"]==item_code]
    if not item_record.empty:
        return item_record["Category Code"].values[0]
    
def get_item_subgroup(item_code: str) -> str:
    item_record = items_info_df[items_info_df["Item Code"]==item_code]
    if not item_record.empty:
        return item_record["Subgroup Code"].values[0]