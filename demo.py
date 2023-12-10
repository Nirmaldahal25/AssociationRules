import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


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


st.header("Dataset Representation")
st.write(
    """
    This is the sample of the dataset we are going to use.
"""
)


dataset = get_dataframe("dataset/groceries.csv")
st.dataframe(dataset, use_container_width=True)
st.markdown(
    """
    > Fig: 1
"""
)
st.write(" Rows: %s, Columns: %s" % dataset.shape)


value = apriori(dataset, min_support=0.05, use_colnames=True, verbose=0)
st.write(value)

association = association_rules(value, metric="confidence", min_threshold=0.2)
st.write(association)
