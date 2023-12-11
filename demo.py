import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

INTRODUCTION = "Introduction"
APRIORI = "Apriori Algorithm"
FP_GROWTH = "FP Growth Algorithm"


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
# st.header("Data Mining Mini Project")
nav_value = st.sidebar.radio(
    "Navigations", options=[INTRODUCTION, APRIORI, FP_GROWTH]
)


dataset = get_dataframe("dataset/groceries.csv")

if nav_value == INTRODUCTION:
    st.header("A Visual Exploration of Groceries Market Basket Dataset")
    st.write(
        """
        Market basket analysis is a data mining technique that helps identify products that customers frequently buy together. 
        It uses association rules to find patterns in purchase history and optimize product placement, pricing, and marketing. 
        It can help to improve customer understanding, inventory management, pricing strategies, and sales growth.

    """)
    st.write(""" The following dataset was used:
""")
    
    st.dataframe(dataset, use_container_width=True)
    st.caption(
        "Table 1: Transaction Matrix of the dataset"
    )
    st.write(
        "Here, ","  Transactions: %s, <br/> &emsp;&emsp;&nbsp;&nbsp;   No. of items: %s" % dataset.shape, unsafe_allow_html=True
    )

    frequency_set = dataset.agg(["sum"]).T
    frequency_set.rename_axis(["Items"], inplace=True)
    st.dataframe(frequency_set, use_container_width=True)
    st.caption(
        "Table 2: Frequency of items in the dataset"
    )

    

elif nav_value in [APRIORI, FP_GROWTH]:
    CONFIDENCE = "confidence"
    LIFT = "lift"
    DEFAULT = "none"
    support = st.sidebar.number_input(
        "Support Threshold", min_value=0.01, max_value=1.0, step=1e-6
    )

    value = pd.DataFrame()
    if nav_value == APRIORI:
        st.header("Apriori Algorithm")
        value = apriori(dataset, min_support=support, use_colnames=True, verbose=0)
        st.write("The Apriori algorithm is a popular data mining technique used for frequent itemset mining and association rule learning over relational databases.", 
                 "The Apriori algorithm has applications in various domains, such as market basket analysis, where it can highlight general trends in the database. <br/><br/> For example, it can help identify items that are often purchased together by customers. This information can then be used to formulate marketing strategies like product placement, promotional offers, etc.", unsafe_allow_html=True)

    elif nav_value == FP_GROWTH:
        st.header("Frequent Pattern Growth Algorithm")
        value = fpgrowth(dataset, min_support=support, use_colnames=True, verbose=0)
        st.write("""The FP-Growth (Frequent Pattern Growth) algorithm is an efficient and scalable method for mining the complete set of frequent patterns in a dataset. 
                 It was proposed as an improvement to the Apriori algorithm to address its inefficiencies.""")

    st.table(value)
    st.write("""Support refers to the relative frequency of an item set in a dataset. 
             It is calculated as the number of transactions containing a particular itemset divided by the total number of transactions. 
             For example, if we have a dataset of 1000 transactions, and the itemset {milk, bread} appears in 100 of those transactions, the support of the itemset {milk, bread} would be 10%. 
             The formula to calculate support is:""")
    st.latex(r"Support(\text{itemset}) = \frac{\text{Number of transactions containing the itemset}}{\text{Total number of transactions}}")
    
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

        if metric == CONFIDENCE:
            st.write("""Confidence is a measure of the reliability or support for a given association rule. 
                     It is defined as the proportion of cases in which the association rule holds true1. 
                     For example, if we have a dataset of 1000 transactions, and the itemset {milk, bread} appears in 100 of those transactions, and the itemset {milk} appears in 200 of those transactions, the confidence of the rule “If a customer buys milk, they will also buy bread” would be 50%. 
                     The formula to calculate confidence is:""")
            st.latex(r"Confidence(\text{rule}) = \frac{\text{Support}(\text{itemset} \cup \text{consequent})}{\text{Support}(\text{itemset})}")

        elif metric == LIFT:
            st.write("""Lift is another important measure that is used to evaluate the performance of a targeting model (association rule) at predicting or classifying cases as having an enhanced response (with respect to the population as a whole), measured against a random choice targeting model.
                        Mathematically, lift is calculated as the ratio of the confidence of the rule to the expected confidence, assuming that the itemsets X and Y are independent of each other. 
                        The formula to calculate lift is:""")
            st.latex(r"Lift(\text{rule}) = \frac{\text{Confidence}(\text{rule})}{\text{Support}(\text{consequent})}")
            st.write("""If the lift is greater than 1, it means that the items in the rule are more likely to be bought together than at random, thus indicating a useful rule. 
                     Conversely, a lift less than 1 indicates that the items are unlikely to be bought together. 
                     A lift of exactly 1 implies that the probability of occurrence of the antecedent and that of the consequent are independent of each other.""")