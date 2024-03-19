import streamlit as st
from efficient_apriori import apriori
import pandas as pd
import base64
import matplotlib.pyplot as plt
import csv
import os

# Function to preprocess data
def preprocess_data(data):
    transactions = [list(row.dropna().astype(str)) for _, row in data.iterrows()]
    return transactions

# Function to save association rules to a CSV file
def save_association_rules(rules, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
        for rule in rules:
            writer.writerow([', '.join(rule.lhs), ', '.join(rule.rhs), rule.support, rule.confidence, rule.lift])

# Function to generate and save plots
def generate_plots(rules, lift_plot_path, scatter_plot_path):
    lifts = [rule.lift for rule in rules]
    confidences = [rule.confidence for rule in rules]
    supports = [rule.support for rule in rules]

    # Lift Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(lifts, bins=20, color='skyblue')
    plt.title('Distribution of Lift')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(lift_plot_path)
    plt.close()

    # Scatter Plot of Confidence vs Support
    plt.figure(figsize=(10, 6))
    plt.scatter(supports, confidences, alpha=0.5, c=lifts, cmap='viridis')
    plt.colorbar(label='Lift')
    plt.title('Confidence vs Support')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.savefig(scatter_plot_path)
    plt.close()

# Function to generate a download link for a dataframe template
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">Download CSV Template</a>'
    return href

# Streamlit UI
st.title('Market Basket Analysis Tool')
st.markdown("## Download CSV Template")
template_df = pd.DataFrame(columns=['Date', 'InvoiceNumber', 'ProductID', 'ProductName', 'Quantity'])
st.markdown(get_table_download_link(template_df), unsafe_allow_html=True)
st.markdown('**Developed by Dr. Sandeep Prasad**')
st.markdown('_Developed as a part of the EY Badge on Retail Sector in the Business Domain_')

# Tutorial or explanation for the analysis
st.markdown("""
This tool helps you perform market basket analysis to discover associations between items. Below are some resources to help you understand the output:
- **Understanding Lift in Association Rules**: Learn about the importance of lift in association rule mining in this [Medium article](https://towardsdatascience.com/understanding-the-lift-in-association-rule-mining-d2b9b9d66e98).
- **Confidence and Support in Market Basket Analysis**: This [tutorial](https://medium.com/analytics-vidhya/association-rules-2-5c2d27e585c2) covers the basics of association rules, including confidence and support.
""")

# Embed a video tutorial
st.video('https://www.youtube.com/watch?v=guVvtZ7ZClw', format='video/mp4', start_time=0)


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
output_dir = st.text_input("Output directory", value="/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/")
sort_criteria = st.multiselect('Sort rules by:', ['confidence', 'support'], default=['confidence'])
top_x_rules = st.number_input('Number of top rules to display:', min_value=1, value=10, step=1)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    transactions = preprocess_data(data)

    min_support = st.slider('Minimum Support', min_value=0.01, max_value=0.5, value=0.02, step=0.01)
    min_confidence = st.slider('Minimum Confidence', min_value=0.01, max_value=1.0, value=0.2, step=0.01)

    if st.button('Analyze'):
        itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
        filtered_rules = [rule for rule in rules if rule.lift >= 1.5]

        # User options for sorting and selecting the number of rules
        sort_by = st.selectbox('Sort rules by:', ['confidence', 'support'])
        top_x_rules = st.number_input('Number of rules to display:', min_value=1, value=10, step=1)

        # Sort the rules based on user selection
        if sort_by == 'confidence':
            filtered_rules.sort(key=lambda rule: rule.confidence, reverse=True)
        elif sort_by == 'support':
            filtered_rules.sort(key=lambda rule: rule.support, reverse=True)
        
        # Limit the number of rules to display
        filtered_rules = filtered_rules[:top_x_rules]

        st.write(f"Displaying top {top_x_rules} rules sorted by {sort_by}:")
        for rule in filtered_rules:
            st.write(f"{rule}")
            
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save association rules to CSV
        csv_file_path = os.path.join(output_dir, 'association_rules.csv')
        save_association_rules(filtered_rules, csv_file_path)
        st.write(f"Association rules saved to {csv_file_path}")

        # Generate and save plots
        lift_plot_path = os.path.join(output_dir, 'lift_distribution.png')
        scatter_plot_path = os.path.join(output_dir, 'confidence_vs_support.png')
        generate_plots(filtered_rules, lift_plot_path, scatter_plot_path)
        st.image(lift_plot_path, caption='Distribution of Lift')
        st.image(scatter_plot_path, caption='Confidence vs Support')

        # After generating and displaying plots
        st.header('Understanding the Graphs')
        st.markdown("""
        ### Distribution of Lift
        The 'Distribution of Lift' graph helps you understand the strength of an association between item sets. A higher lift value means a stronger association. It's crucial for identifying the most significant rules generated by the analysis.

        ### Confidence vs Support
        The 'Confidence vs Support' scatter plot visualizes the reliability (confidence) and frequency (support) of the rules. Points closer to the top-right corner represent rules that are both frequent and reliable.

        For a deeper dive into how to interpret these graphs and apply the insights to your retail strategy, check out the following resources:
        - [Market Basket Analysis: Identifying Products and Content That Go Well Together](https://towardsdatascience.com/market-basket-analysis-978ac064d8c6)
        - [Video Tutorial on Market Basket Analysis and Association Rules](https://www.youtube.com/watch?v=guVvtZ7ZClw)
        """)