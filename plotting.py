import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

def count_merchant_types(transactions_log):
    """
    Count the occurrences of each 'Merchant Type'.

    Parameters:
    transactions_log (pd.DataFrame): DataFrame containing transactions log.

    Returns:
    pd.Series: Series containing the count of each merchant type.
    """
    return transactions_log['merchant_category'].value_counts()

def filter_agent_data(agent_id, transactions_log, account_state_log):
    """
    Filter data for the specified agent.

    Parameters:
    agent_id (int): The ID of the agent to filter.
    transactions_log (pd.DataFrame): DataFrame containing transactions log.
    account_state_log (pd.DataFrame): DataFrame containing account state logs.

    Returns:
    tuple: Filtered transactions and account state logs for the agent.
    """
    agent_transactions = transactions_log[transactions_log['agent_id'] == agent_id]
    agent_account_state = account_state_log[account_state_log['agent_id'] == agent_id]
    return agent_transactions, agent_account_state


def plot_transaction_amount_distribution(agent_transactions, agent_id=None):
    """
    Plot histogram of transaction amounts.

    Parameters:
    agent_transactions (pd.DataFrame): DataFrame containing agent transactions.
    agent_id (int, optional): The ID of the agent. If None, plot for the entire population.
    """
    if agent_id is not None:
        agent_transactions = agent_transactions[agent_transactions['agent_id'] == agent_id]
        title = f'Agent {agent_id} Transaction Amount Distribution'
    else:
        title = 'Population Transaction Amount Distribution'
    
    plt.figure(figsize=(12, 6))
    sns.histplot(agent_transactions['amount'], bins=30, kde=True)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def plot_spending_patterns(agent_id, agent_transactions):
    """
    Plot heatmap of spending patterns across different merchant categories.

    Parameters:
    agent_id (int): The ID of the agent.
    agent_transactions (pd.DataFrame): DataFrame containing agent transactions.
    """
    merchant_category_summary = agent_transactions.groupby('merchant_category').agg(
        total_spending=('amount', 'sum'),
        avg_transaction_amount=('amount', 'mean'),
        transaction_count=('amount', 'count')
    ).reset_index()

    merchant_category_pivot = merchant_category_summary.pivot(index='merchant_category', columns='transaction_count', values='total_spending').fillna(0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(merchant_category_pivot, cmap='YlGnBu', linewidths=0.5)
    plt.xlabel('Transaction Count')
    plt.ylabel('Merchant Category')
    plt.title(f'Agent {agent_id} Spending Patterns Across Different Merchant Categories')
    plt.show()

def plot_merchant_type_distribution(transactions_log):
    """
    Plot the distribution of 'Merchant Type'.

    Parameters:
    transactions_log (pd.DataFrame): DataFrame containing transactions log.
    """

    merchant_type_counts = count_merchant_types(transactions_log)

    plt.figure(figsize=(10, 6))
    merchant_type_counts.plot(kind='bar')  # Creates a bar chart
    plt.title('Distribution of Merchant Types')
    plt.xlabel('Merchant Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha="right")  # Rotate the x-axis labels for better readability
    plt.tight_layout(pad=3)  # Adjust the layout to make room for the rotated labels
    plt.show()


def plot_event_distribution(log, column, title, xlabel, bins=100, alpha=0.5):
    """
    Plot the distribution of events for a specified column.

    Parameters:
    log (pd.DataFrame): DataFrame containing the log (transactions or payments).
    column (str): The column to count events.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    bins (int): Number of bins for the histogram.
    alpha (float): Transparency level for the histogram.
    """
    event_counts = log[column].value_counts()
    plt.figure(figsize=(10, 6))
    event_counts.plot(kind='hist', bins=bins, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def categorize_status(status):
    """
    Categorize account statuses into 'Good', 'Delinquent', and 'Charge Off'.

    Parameters:
    status (str): The account status.

    Returns:
    str: The categorized status.
    """
    if status == 'charge off':
        return 'Charge Off'
    elif status == 'delinquent':
        return 'Delinquent'
    else:
        return 'Good'


def plot_behavior_over_time(log, agent_ids=None, sample_size=None, period='W', data_type='transactions'):
    """
    Plot behavior over time for different agents, aggregated over specified periods.

    Parameters:
    log (pd.DataFrame): DataFrame containing the log (transactions or payments).
    agent_ids (list): List of agent IDs to include in the plot. If None, include all agents.
    sample_size (int): Number of agents to sample. If None, include all agents.
    period (str): Resampling period ('W' for weekly, 'M' for monthly).
    data_type (str): Type of data ('transactions' or 'payments').
    """
    if agent_ids is not None:
        log = log[log['agent_id'].isin(agent_ids)]
    if sample_size is not None:
        agent_ids = log['agent_id'].unique()
        sampled_agent_ids = np.random.choice(agent_ids, size=sample_size, replace=False)
        log = log[log['agent_id'].isin(sampled_agent_ids)]
    
    plt.figure(figsize=(12, 6))
    for agent_id in log['agent_id'].unique():
        agent_log = log[log['agent_id'] == agent_id]
        agent_log.set_index('timestamp', inplace=True)
        agent_log_resampled = agent_log['amount'].resample(period).sum()
        plt.plot(agent_log_resampled.index, agent_log_resampled, label=f'Agent {agent_id}')
    
    plt.xlabel('Timestamp')
    plt.ylabel('Amount')
    plt.title(f'{data_type.capitalize()} Behavior Over Time for Different Agents (Aggregated by {period})')
    plt.legend()
    plt.show()


# Display the first few rows and data types of each table
def display_table_info(df, table_name):
    print(f"Schema and attributes of {table_name}:")
    print(df.head(), "\n")
    print(df.info(), "\n")