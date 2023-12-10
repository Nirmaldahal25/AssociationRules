import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

DATATASET_SAMPLE = "Dataset Sample"
APRIORI = "Apriori"
FP_GROWTH = "FP Growth"


@st.cache_data
def get_dataframe(filename):
    transactions = []
    with open(filename, "r") as file:
        for line in file:
            transaction = line.strip().split(",")
            transaction_dict = {item: 1 for item in transaction}
            transactions.append(transaction_dict)

    # create a transactional matrix
    dataframe = pd.DataFrame(transactions)
    dataframe.fillna(value=0, inplace=True)
    return dataframe


# header
st.header("Data Mining Mini Project")
nav_value = st.sidebar.radio(
    "Navigations", options=[DATATASET_SAMPLE, APRIORI, FP_GROWTH]
)


dataset = get_dataframe("dataset/groceries.csv")

if nav_value == DATATASET_SAMPLE:
    st.header("A Visual Exploration of Groceries Market Basket Dataset")
    st.write(
        """
        Market basket analysis is a data mining technique that helps identify products that customers frequently buy together. 
        It uses association rules to find patterns in purchase history and optimize product placement, pricing, and marketing. 
        It can help to improve customer understanding, inventory management, pricing strategies, and sales growth.

    """
    )
    st.dataframe(dataset, use_container_width=True)
    st.markdown(
        """
        > Fig: 1
    """
    )
    st.write(" Rows: %s, Columns: %s" % dataset.shape)

elif nav_value in [APRIORI, FP_GROWTH]:
    CONFIDENCE = "confidence"
    LIFT = "lift"
    DEFAULT = "none"
    support = st.sidebar.number_input(
        "Support Threshold", min_value=0.01, max_value=1.0, step=1e-6
    )

    value = pd.DataFrame()
    if nav_value == APRIORI:
        st.subheader("Apriori Algorithm")
        value = apriori(dataset, min_support=support, use_colnames=True, verbose=0)

    elif nav_value == FP_GROWTH:
        st.subheader("FPgrowth Algorithm")
        value = fpgrowth(dataset, min_support=support, use_colnames=True, verbose=0)

    st.table(value)
    metric = st.sidebar.selectbox("Metrics", [DEFAULT, CONFIDENCE, LIFT])
    threshold = 0
    if metric == CONFIDENCE:
        threshold = st.sidebar.number_input(
            f"Threshold {metric}", min_value=0.0, max_value=1.0, step=1e-3
        )
    elif metric == LIFT:
        threshold = st.sidebar.number_input(
            f"Threshold {metric}", min_value=0.0, step=1e-3
        )

    if metric != DEFAULT:
        st.subheader("Metrics")
        association = association_rules(value, metric=metric, min_threshold=threshold)
        st.write(association)
