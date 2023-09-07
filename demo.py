import os
import streamlit as st
import pandas as pd
from recommendation_module import *

@st.cache_data
def load_model(model_name):
    return Item2Item(load_dir=model_name)

@st.cache_data
def load_items():
    df = pd.read_excel("data/products info.xlsx")
    df = df[["Item Code", "Item", "Product Size"]]
    df["Item Code"] = df["Item Code"].astype(str)
    df.set_index("Item Code", inplace=True)
    return df.to_dict(orient="index")

available_models = os.listdir("models/item2item")
c1, c2 = st.columns(2)
with c1:
    model_name = st.selectbox('Select Model', available_models, index=1)
with c2:
    st.text(" ")
    st.text(" ")
    exclude_subgroup = st.checkbox("Exclude Subgroup")

model = load_model(model_name)
items_mapper = load_items()
item_code = st.text_input("Enter Item Code")
if item_code:
    item_entry = items_mapper[item_code]
    text = item_entry["Item"] + " (" + str(item_entry["Product Size"]) + ")"
    st.markdown(f"<h2 style='text-align: center;' dir='rtl'>{text}</h2>", unsafe_allow_html=True)
    results = model.get_top_n_frequent_items(item_code, exclude_subgroup=exclude_subgroup)

    cols = st.columns(len(results["items"]))
    for i in range(len(cols)):
        with cols[i]:
            item_entry = items_mapper[results["items"][i]]
            text = item_entry["Item"] + " (" + str(item_entry["Product Size"]) + ")"
            score = results["scores"][i]
            st.markdown(f"<p style='text-align: center;' dir='rtl'>{text}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;' dir='rtl'>Score: {score:.3f}</p>", unsafe_allow_html=True)
