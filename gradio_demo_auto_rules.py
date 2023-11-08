import gradio as gr
import plotly.express as px
import os
import pandas as pd
from recommendation_module import *

def load_items():
    df = pd.read_excel("data/products info (shortlist).xlsx")
    df = df[["Item Code", "Item", "Product Size"]]
    df["Item Code"] = df["Item Code"].astype(str)
    df.set_index("Item Code", inplace=True)
    return df.to_dict(orient="index")

def bar_plot_fn(item_code, n):
    output = model.get_top_n_frequent_items(item_code, n=100)
    output.apply_category_filters(item_code)
    output = output.get_top_n_recommendations(n=int(n))
    df = pd.DataFrame(
            {
            "items": [items_mapper[i_code]["Item"] + " (" + str(items_mapper[i_code]["Product Size"]) + ")" for i_code in output["items"]],
            "scores": output["scores"]
            })
    fig = px.bar(df, x="scores", y="items", orientation="h")
    fig.update_layout(yaxis=dict(tickfont=dict(size=16)), height=300)
    return fig, gr.Textbox.update(items_mapper[item_code]["Item"] + " (" + str(items_mapper[item_code]["Product Size"]) + ")")


#available_models = os.listdir("models/item2item")
model = Item2Item(model_name="Loyalty-03-10-2023")
items_mapper = load_items()
#items_mapper = utils.items_info_df.set_index("Item Code")[["Item", "Product Size"]].to_dict(orient="index")

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            item_code = gr.Textbox(label="Item Code")
            num_of_recommendations = gr.Number(5, label="Number of Recommendations")
        with gr.Row():
            submit = gr.Button()
        prod_name = gr.Textbox(label="Item Name")
        plot = gr.Plot()

    submit.click(bar_plot_fn, inputs=[item_code, num_of_recommendations], outputs=[plot, prod_name])

demo.launch()
