import streamlit as st
from efficient_apriori import apriori
import pandas as pd
import base64
import matplotlib.pyplot as plt
import csv
import os
import time  # Import the time module


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
def generate_plots(rules, lift_plot_path, scatter_plot_path, color_scheme='skyblue', cmap='viridis'):
    lifts = [rule.lift for rule in rules]
    confidences = [rule.confidence for rule in rules]
    supports = [rule.support for rule in rules]

    # Lift Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(lifts, bins=20, color=color_scheme)  # Use the selected color scheme
    plt.title('Distribution of Lift')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(lift_plot_path)
    plt.close()

    # Scatter Plot of Confidence vs Support
    plt.figure(figsize=(10, 6))
    plt.scatter(supports, confidences, alpha=0.5, c=lifts, cmap=cmap)  # Use the user-selected colormap
    plt.colorbar(label='Lift')
    plt.title('Confidence vs Support')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.savefig(scatter_plot_path)
    plt.close()

# Function to generate a download link for a dataframe template
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">Download CSV Template</a>'
    return href

# Function to generate a download link for files
def get_download_link(file_path, file_label, file_type='file/csv'):
    with open(file_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:{file_type};base64,{b64}" download="{file_path.split("/")[-1]}">Download {file_label}</a>'
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

# Dynamic output directory selection
default_dir = "/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/"
use_default_dir = st.checkbox("Use default output directory", value=True)
if use_default_dir:
    output_dir = default_dir
else:
    output_dir = st.text_input("Specify output directory", value=default_dir)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
sort_criteria = st.multiselect('Sort rules by:', ['confidence', 'support'], default=['confidence'])
top_x_rules = st.number_input('Number of top rules to display:', min_value=1, value=10, step=1)

# Color scheme selection
# Place this section in your Streamlit UI code, before the analysis button
color_scheme = st.selectbox('Choose a color for the histogram:', ['skyblue', 'lightgreen', 'plum', 'coral'])
cmap = st.selectbox('Choose a colormap for the scatter plot:', ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    transactions = preprocess_data(data)

    min_support = st.slider('Minimum Support', min_value=0.01, max_value=0.5, value=0.02, step=0.01)
    min_confidence = st.slider('Minimum Confidence', min_value=0.01, max_value=1.0, value=0.2, step=0.01)

    if st.button('Analyze'):
        with st.spinner('Analyzing... Please wait.'):
            # Initialize the progress bar here
            progress_bar = st.progress(0)

            # Perform the actual analysis
            itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
            filtered_rules = [rule for rule in rules if rule.lift >= 1.5]
            
            # Simulate a long-running analysis with incremental updates to the progress bar
            for i in range(100):
                # Assuming each iteration represents a step in the analysis
                progress_bar.progress(i + 1)
                time.sleep(0.01)  # Simulate time taken for each step of the analysis
            
            # You might want to update the progress bar to complete at the end of analysis
            progress_bar.progress(100)

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


        # Display the results
        st.write(f"Displaying top {len(filtered_rules)} rules:")
        for rule in filtered_rules:
            st.write(f"{rule}")

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

        # Generate and save plots with the selected color scheme
        generate_plots(filtered_rules, lift_plot_path, scatter_plot_path, color_scheme=color_scheme, cmap=cmap)
        st.image(lift_plot_path, caption='Distribution of Lift')
        st.image(scatter_plot_path, caption='Confidence vs Support')

        # Display download links for the generated files
        csv_download_link = get_download_link(csv_file_path, "Association Rules CSV", "text/csv")
        lift_plot_download_link = get_download_link(lift_plot_path, "Lift Distribution Plot", "image/png")
        scatter_plot_download_link = get_download_link(scatter_plot_path, "Confidence vs Support Plot", "image/png")

        st.markdown(csv_download_link, unsafe_allow_html=True)
        st.markdown(lift_plot_download_link, unsafe_allow_html=True)
        st.markdown(scatter_plot_download_link, unsafe_allow_html=True)

        # Interactive analysis results
        st.header('Analysis Results')
        st.dataframe(filtered_rules)  # Assuming filtered_rules is a DataFrame

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