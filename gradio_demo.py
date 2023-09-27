import gradio as gr
import plotly.express as px
import os
import pandas as pd
from recommendation_module import *
import utils

def load_items():
    df = pd.read_excel("data/products info (shortlist).xlsx")
    df = df[["Item Code", "Item", "Product Size"]]
    df["Item Code"] = df["Item Code"].astype(str)
    df.set_index("Item Code", inplace=True)
    return df.to_dict(orient="index")

def bar_plot_fn(item_code, exclude_subgroup, exclude_product_group, items_per_subgroup_limit, item_categories, n, min_support, min_confidence, min_lift):
    output = model.get_top_n_frequent_items(item_code, n=int(n), exclude_subgroup=False, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
    output = utils.post_ranking(item_code, output, exclude_subgroup, exclude_product_group, item_categories)
    if items_per_subgroup_limit:
        output = utils.max_items_per_subgroup(model_out=output, n=1)
    df = pd.DataFrame(
            {
            "items": [items_mapper[i_code]["Item"] + " (" + str(items_mapper[i_code]["Product Size"]) + ")" for i_code in output["items"]],
            "scores": output["scores"]
            })
    fig = px.bar(df, x="scores", y="items", orientation="h")
    fig.update_layout(yaxis=dict(tickfont=dict(size=16)), height=300)
    return fig, gr.Textbox.update(items_mapper[item_code]["Item"] + " (" + str(items_mapper[item_code]["Product Size"]) + ")")


#available_models = os.listdir("models/item2item")
model = Item2Item(load_dir="1-year-V02")
items_mapper = load_items()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            num_of_recommendations = gr.Number(5, label="Number of Recommendations")
            min_support = gr.Number(0.0001, label="min_support", step=0.05)
            min_confidence = gr.Number(0.01, label="min_confidence", step=0.05)
            min_lift = gr.Number(1, label="min_lift", step=0.05)
        with gr.Row():
            item_code = gr.Textbox(label="Item Code")
            with gr.Column():
                exclude_subgroup = gr.Checkbox(label="Exclude Subgroup")
                exclude_product_group = gr.Checkbox(label="Exclude Product Group")
                items_per_subgroup_limit = gr.Checkbox(label="Items Per Subgroup Limit")
        unique_categories = utils.items_info_df["Category Code"].unique().tolist()
        item_categories = gr.Dropdown(choices=unique_categories, value=unique_categories, label="Item Category", multiselect=True)
        with gr.Row():
            submit = gr.Button()
        prod_name = gr.Textbox(label="Item Name")
        plot = gr.Plot()

    submit.click(bar_plot_fn, inputs=[item_code, exclude_subgroup, exclude_product_group, items_per_subgroup_limit, item_categories, num_of_recommendations, min_support, min_confidence, min_lift], outputs=[plot, prod_name])

demo.launch()
