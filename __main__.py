import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle




PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.


def main(test_set_dir: str, results_dir: str):
    
    # Load test set data.
    account_state_df = pd.read_csv(os.path.join(test_set_dir, "account_state_log.csv"))
    payments_df = pd.read_csv(os.path.join(test_set_dir, "payments_log.csv"), parse_dates=['timestamp'])
    transactions_df = pd.read_csv(os.path.join(test_set_dir, "transactions_log.csv"), parse_dates=['timestamp'])

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.
    aggregated = account_state_df.groupby('agent_id').agg({
    'credit_balance': ['mean', 'max', 'last'],  # Average, peak, and final balance
    'credit_utilization': ['mean', 'max', 'last'],  # Utilization stats
    'interest_rate': ['mean', 'last'],  # Average and final interest rate
    'current_missed_payments': ['sum', 'max', 'last'],  # Total, peak, and latest missed payments
    'timestamp': ['min', 'max']  # First and last timestamps
    })

    # Flatten MultiIndex columns for easier use
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
    aggregated.reset_index(inplace=True)

    # Group by `agent_id` and aggregate
    aggregated_transactions = transactions_df.groupby('agent_id').agg({
        'amount': ['sum', 'mean', 'max', 'count'],  # Transaction value stats
        'status': lambda x: (x == 'completed').sum(),  # Count of completed transactions
        'merchant_category': lambda x: x.mode()[0] if not x.mode().empty else None,  # Most frequent category
        'merchant_id': pd.Series.nunique,  # Unique merchants
        'online': ['sum', 'mean'],  # Total and proportion of online transactions
        'timestamp': ['min', 'max']  # Transaction timeline
    })

    # Flatten MultiIndex columns
    aggregated_transactions.columns = ['_'.join(col).strip() for col in aggregated_transactions.columns]
    aggregated_transactions.reset_index(inplace=True)

    # Rename for clarity
    aggregated_transactions.rename(columns={
        'amount_sum': 'total_transaction_amount',
        'amount_mean': 'average_transaction_amount',
        'amount_max': 'max_transaction_amount',
        'amount_count': 'total_transactions',
        'status_<lambda>': 'completed_transactions',
        'merchant_id_nunique': 'unique_merchants',
        'online_sum': 'online_transactions',
        'online_mean': 'online_transaction_proportion',
        'timestamp_min': 'first_transaction_date',
        'timestamp_max': 'last_transaction_date'
    }, inplace=True)

    transactions_df = pd.get_dummies(transactions_df, columns=['merchant_category'], prefix='', prefix_sep='')

    # Now, perform your aggregation as before
    unique_categories = ['Misc', 'Retail', 'Business', 'Clothing', 'Agricultural', 
                            'Contractor', 'Transportation', 'Utility', 'Professional']

    # Group by `agent_id` and aggregate other columns
    aggregated_transactions = transactions_df.groupby('agent_id').agg({
        'amount': ['sum', 'mean', 'max', 'count'],  # Transaction value stats
        'online': ['sum', 'mean'],  # Total and proportion of online transactions
        'timestamp': ['min', 'max'],  # Transaction timeline
    }).reset_index()

    # Aggregate the merchant category counts (sum the binary columns)
    for category in unique_categories:
        aggregated_transactions[category] = transactions_df.groupby('agent_id')[category].sum().reset_index(drop=True)

    # Flatten multi-level columns in the aggregated DataFrame
    aggregated_transactions.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in aggregated_transactions.columns]

    # Add custom metrics for `status` outside of the main aggregation
    status_counts = transactions_df.groupby('agent_id')['status'].value_counts().unstack(fill_value=0).reset_index()
    status_counts.rename(columns={'approved': 'approved_count', 'declined': 'declined_count'}, inplace=True)

    # Merge the status counts back with the aggregated data
    aggregated_transactions = pd.merge(aggregated_transactions, status_counts, on='agent_id', how='left')

    # Drop the `merchant_id` column if it exists (optional, as per your requirement)
    aggregated_transactions = aggregated_transactions.drop(columns=[col for col in aggregated_transactions.columns if 'merchant_id' in col], errors='ignore')

        # Group by `agent_id` and aggregate
    aggregated_payments = payments_df.groupby('agent_id').agg({
        'amount': ['sum', 'mean', 'max', 'min', 'count', 'last'],  # Payment stats
        'timestamp': ['min', 'max']  # Payment timeline
    })

    # Flatten MultiIndex columns
    aggregated_payments.columns = ['_'.join(col).strip() for col in aggregated_payments.columns]
    aggregated_payments.reset_index(inplace=True)

    # Add derived features (e.g., payment frequency, time difference)
    aggregated_payments['payment_frequency'] = aggregated_payments['amount_count'] / (
        (aggregated_payments['timestamp_max'] - aggregated_payments['timestamp_min']).dt.days + 1
    )  # Payments per day
    aggregated_payments['time_between_first_last'] = (
        aggregated_payments['timestamp_max'] - aggregated_payments['timestamp_min']
    ).dt.days    
    # In lieu of doing something test inputs, maybe you "learned" from training data that
    #  30% of accounts are charge-off across all periods (not true), so you randomly 
    #  guess with that percentage.
    co_percent = 0.3
    agents = list(set(account_state_df.agent_id).union(set(payments_df.agent_id)).union(set(transactions_df.agent_id)))
    col_names = {months: f"charge_off_within_{months}_months" for months in PREDICTION_WINDOW_MONTHS}
    output_df = pd.DataFrame(columns=["agent_id"] + list(col_names.values()))
    output_df["agent_id"] = agents
    ##############################
    merged_df = output_df.merge(aggregated, on='agent_id', how='left')
    # Convert timestamp columns to datetime
    merged_df['timestamp_min'] = pd.to_datetime(merged_df['timestamp_min'])
    merged_df['timestamp_max'] = pd.to_datetime(merged_df['timestamp_max'])

    # Calculate the difference in days
    merged_df['timestamp_diff'] = (merged_df['timestamp_max'] - merged_df['timestamp_min']).dt.days

    # Drop the original timestamp columns
    merged_df.drop(columns=['timestamp_min', 'timestamp_max'], inplace=True)

    # Display the updated DataFrame
    merged_df.head()
    merged_df2 = merged_df.merge(aggregated_payments, on='agent_id', how='left')
    merged_df2['amount_sum'].fillna(0, inplace=True)
    merged_df2['amount_mean'].fillna(0, inplace=True)
    merged_df2['amount_max'].fillna(0, inplace=True)
    merged_df2['amount_min'].fillna(0, inplace=True)
    merged_df2['amount_count'].fillna(0, inplace=True)
    merged_df2['amount_last'].fillna(0, inplace=True)
    merged_df2['payment_frequency'].fillna(0, inplace=True)
    merged_df2.drop(columns=['timestamp_min', 'timestamp_max','time_between_first_last'], inplace=True)
    merged_df3 = merged_df2.merge(aggregated_transactions, on='agent_id', how='left')
    merged_df3['amount_sum_y'].fillna(0, inplace=True)
    merged_df3['amount_mean_y'].fillna(0, inplace=True)
    merged_df3['amount_max_y'].fillna(0, inplace=True)
    merged_df3['amount_count_y'].fillna(0, inplace=True)
    merged_df3['online_sum'].fillna(0, inplace=True)
    merged_df3['online_mean'].fillna('0', inplace=True)
    merged_df3['Misc'].fillna(0, inplace=True)
    merged_df3['Retail'].fillna(0, inplace=True)
    merged_df3['Business'].fillna(0, inplace=True)
    merged_df3['Clothing'].fillna(0, inplace=True)
    merged_df3['Agricultural'].fillna(0, inplace=True)
    merged_df3['Contractor'].fillna(0, inplace=True)
    merged_df3['Transportation'].fillna(0, inplace=True)
    merged_df3['Utility'].fillna(0, inplace=True)
    merged_df3['Professional'].fillna(0, inplace=True)
    merged_df3['approved_count'].fillna(0, inplace=True)
    merged_df3['declined_count'].fillna(0, inplace=True)
    merged_df3.drop(columns=['timestamp_min', 'timestamp_max'], inplace=True)
    merged_final=merged_df3.copy()
    merged_final['online_mean']=merged_final['online_mean'].astype('float64')
    with open(os.path.join(test_set_dir,'model1.pkl'), 'rb') as f:
        clf2 = pickle.load(f)
    month3=clf2.predict(merged_final[['current_missed_payments_last', 'current_missed_payments_sum',
       'current_missed_payments_max', 'Misc', 'amount_count_y',
       'timestamp_diff', 'Agricultural', 'Clothing', 'online_sum',
       'Transportation']])
    with open(os.path.join(test_set_dir,'model2.pkl'), 'rb') as f:
        clf2 = pickle.load(f)
    month6=clf2.predict(merged_final[['current_missed_payments_last', 'current_missed_payments_sum',
       'current_missed_payments_max', 'Misc', 'online_sum', 'amount_count_y',
       'Clothing', 'timestamp_diff', 'credit_utilization_last',
       'Agricultural']])
    with open(os.path.join(test_set_dir,'model3.pkl'), 'rb') as f:
        clf2 = pickle.load(f)
    month9=clf2.predict(merged_final[['current_missed_payments_last', 'current_missed_payments_max',
       'current_missed_payments_sum', 'Misc', 'timestamp_diff',
       'amount_count_y', 'online_sum', 'Clothing', 'Agricultural',
       'credit_utilization_max']])
    with open(os.path.join(test_set_dir,'model4.pkl'), 'rb') as f:
        clf2 = pickle.load(f)
    month12=clf2.predict(merged_final[['Misc', 'timestamp_diff', 'current_missed_payments_last',
       'current_missed_payments_max', 'current_missed_payments_sum',
       'amount_count_y', 'online_sum', 'Transportation', 'approved_count',
       'Professional']])


    ##############################
    for months in PREDICTION_WINDOW_MONTHS:
        col_name = col_names[months]
        preds = month3 if months == 3 else month6 if months == 6 else month9 if months == 9 else month12
        output_df[col_name] = preds

    # END PROCESSING TEST SET INPUTS
    # ---------------------------------

    # NOTE: name "results.csv" is a must.
    output_df.to_csv(os.path.join(results_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bth_test_set",
        type=str,
        required=True
    )
    parser.add_argument(
        "--bth_results",
        type=str,
        required=True
    )

    args = parser.parse_args()
    main(args.bth_test_set, args.bth_results)