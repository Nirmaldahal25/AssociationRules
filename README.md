# Market Basket Analysis with Apriori and FP-Growth

## Overview

This repository explores Market Basket Analysis (MBA) using the Apriori and FP-Growth algorithms. The analysis is applied to a grocery dataset, aiming to identify patterns and association rules in customer purchase behavior.


## Introduction

Market Basket Analysis is a data mining technique that helps businesses understand customer purchasing patterns. This project utilizes two widely used algorithms, Apriori and FP-Growth, to extract meaningful insights from a grocery dataset.

### Apriori

The Apriori algorithm, proposed by R. Agrawal and R. Srikant in 1994, is employed for frequent itemset mining and association rule learning. It reveals general trends in the dataset by iteratively scanning and building candidate sets.

### FP-Growth

The FP-Growth algorithm uses a tree structure (FP-tree) to efficiently mine frequent itemsets. It stores all transactions in a trie data structure, allowing for faster processing compared to Apriori in certain scenarios.

## Streamlit Visualization

Streamlit is integrated into the project for dynamic and interactive data visualization. Specific visualizations, such as sactter plot, line graph, aid in the interpretation of market basket analysis results.


## Getting Started

To run the code locally, follow these steps:

1. Clone the repository.
2. Setting Up a Virtual Environment and Installing Dependencies:
   
   ```bash
   # Create a Virtual Environment
   python3 -m venv venv
   
   # Activate the Virtual Environment (Linux/macOS)
   source venv/bin/activate
   
   # Install Packages from requirements.txt
   pip install -r requirements.txt
   
4. Run the following command:
  
   ```python
   streamlit run demo.py
