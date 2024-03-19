import pandas as pd
from efficient_apriori import apriori
import csv
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import plotly.express as px

def get_user_input():
    print("\nWelcome to the Market Basket Analysis Tool!")
    print("This tool helps you discover interesting relationships between items in a large dataset.")
    while True:
        try:
            min_support = float(input("\nEnter minimum support value (e.g., 0.02 for 2%): "))
            min_confidence = float(input("Enter minimum confidence value (e.g., 0.01 for 1%): "))
            print("\nAnalyzing data with the following parameters:")
            print(f"Minimum Support: {min_support}")
            print(f"Minimum Confidence: {min_confidence}")
            return min_support, min_confidence
        except ValueError:
            print("Please enter a valid numeric value.")

def preprocess_data(filename):
    print("\nPreprocessing data...")
    data = pd.read_csv(filename, low_memory=False)
    transactions = [list(row.dropna().astype(str)) for _, row in data.iterrows()]
    print("Data preprocessing complete.")
    return transactions

def main():
    filename = '/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/market_basket_transactions.csv'
    transactions = preprocess_data(filename)
    min_support, min_confidence = get_user_input()
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)

    print("\nGenerating association rules...")
    min_lift = 1.5
    filtered_rules = [rule for rule in rules if rule.lift >= min_lift]
    print(f"Found {len(filtered_rules)} rules that meet the criteria.")

    save_association_rules(filtered_rules)
    generate_plots(filtered_rules)
    save_pretty_table(filtered_rules)
    print("\nAnalysis complete. Check the output files for detailed results.")

def save_association_rules(rules):
    csv_file_path = '/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/association_rules.csv'
    print(f"\nSaving association rules to {csv_file_path}...")
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
        for rule in rules:
            writer.writerow([', '.join(rule.lhs), ', '.join(rule.rhs), rule.support, rule.confidence, rule.lift])
    print("Association rules saved successfully.")

def generate_plots(rules):
    print("\nGenerating and saving plots...")
    # Calculate lifts here so it can be used in both plotting functions
    lifts = [rule.lift for rule in rules]
    generate_lift_distribution_plot(rules, lifts)
    generate_confidence_vs_support_plot(rules, lifts)
    print("Plots saved successfully.")

def generate_lift_distribution_plot(rules, lifts):
    plt.figure(figsize=(10, 6))
    plt.hist(lifts, bins=20, color='skyblue')
    plt.title('Distribution of Lift')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/lift_distribution.png')
    plt.close()

def generate_confidence_vs_support_plot(rules, lifts):
    confidences = [rule.confidence for rule in rules]
    supports = [rule.support for rule in rules]
    plt.figure(figsize=(10, 6))
    plt.scatter(supports, confidences, alpha=0.5, c=lifts, cmap='viridis')
    plt.colorbar(label='Lift')
    plt.title('Confidence vs Support')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.savefig('/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/confidence_vs_support.png')
    plt.close()

def save_pretty_table(rules):
    table_file_path = '/Users/sandeepprasad/Documents/1. EY_Bronze Badge_Retail/association_rules_table.txt'
    print(f"\nSaving pretty table of rules to {table_file_path}...")
    table = PrettyTable()
    table.field_names = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
    for rule in sorted(rules, key=lambda rule: rule.lift, reverse=True):
        table.add_row([', '.join(rule.lhs), ', '.join(rule.rhs), round(rule.support, 4), round(rule.confidence, 4), round(rule.lift, 4)])
    with open(table_file_path, 'w') as file:
        file.write(table.get_string())
    print("Pretty table saved successfully.")

if __name__ == "__main__":
    main()
