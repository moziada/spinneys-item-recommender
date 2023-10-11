import pandas as pd

class RecommendationResult:
    def __init__(self, items: list, scores: list, items_info_path: str):
        self.items = items
        self.scores = scores
        self.items_info_df = pd.read_parquet(items_info_path, dtype=str)

    def rule_max_items_per_subgroup(self, n: int=1):
        subgroup_count = {}
        for i, item_code in enumerate(self.items):
            subgroup_code = self.get_item_subgroup(item_code)
            if subgroup_count.get(subgroup_code, 0) >= n:
                self.items.pop(i)
                self.scores.pop(i)
            subgroup_count[subgroup_code] = subgroup_count.get(subgroup_code, 0) + 1

    def rule_same_category(self, antecedent_item_category: str):
        for i, item_code in enumerate(self.items):
            consequen_item_category = self.get_item_category(item_code)
            if antecedent_item_category != consequen_item_category:
                self.items.pop(i)
                self.scores.pop(i)

    def rule_different_category(self, antecedent_item_category: str):
        for i, item_code in enumerate(self.items):
            consequen_item_category = self.get_item_category(item_code)
            if antecedent_item_category == consequen_item_category:
                self.items.pop(i)
                self.scores.pop(i)

    def rule_different_subgroup(self, antecedent_item_subgroup: str):
        for i, item_code in enumerate(self.items):
            consequen_item_subgroup = self.get_item_subgroup(item_code)
            if antecedent_item_subgroup == consequen_item_subgroup:
                self.items.pop(i)
                self.scores.pop(i)

    def apply_category_filters(self, antecedent_item_code):
        antecedent_item_category = self.get_item_category(antecedent_item_code)
        # H&B category = 0104, GM division categories = 03xx
        if antecedent_item_category == "0104" or antecedent_item_category.startswith("03"):
            self.rule_same_category(antecedent_item_category)
        
        # if category is meat, recommend products out of meat category
        elif antecedent_item_category == "0209":
            self.rule_different_category("0209")
        
        # if nothing from above, just remove recommendations from the same subgroup
        else:
            antecedent_item_subgroup = self.get_item_subgroup(antecedent_item_code)
            self.rule_different_subgroup(antecedent_item_subgroup)
        
        self.rule_max_items_per_subgroup(n=1)
        
    def get_item_category(self, item_code: str) -> str:
        item_record = self.items_info_df[self.items_info_df["Item Code"]==item_code]
        if not item_record.empty:
            return item_record["Category Code"].values[0]
        
    def get_item_subgroup(self, item_code: str) -> str:
        item_record = self.items_info_df[self.items_info_df["Item Code"]==item_code]
        if not item_record.empty:
            return item_record["Subgroup Code"].values[0]
    
    def get_top_n_recommendations(self, n: int):
        n = min(n, len(self.items))
        return {"items": self.items[:n], "scores": self.scores[:n]}
