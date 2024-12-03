import argparse
import os

import pandas as pd
import numpy as np


PREDICTION_WINDOW_MONTHS = [3, 6, 9, 12]  # Constant for this charge-off prediction task.


def main(test_set_dir: str, results_dir: str):
    
    # Load test set data.
    account_state_df = pd.read_csv(os.path.join(test_set_dir, "account_state_log.csv"))
    payments_df = pd.read_csv(os.path.join(test_set_dir, "payments_log.csv"))
    transactions_df = pd.read_csv(os.path.join(test_set_dir, "transactions_log.csv"))

    # ---------------------------------
    # START PROCESSING TEST SET INPUTS
    # Beep boop bop you should do something with test inputs unlike this script.

    # In lieu of doing something test inputs, maybe you "learned" from training data that
    #  30% of accounts are charge-off across all periods (not true), so you randomly 
    #  guess with that percentage.
    co_percent = 0.3
    agents = list(set(account_state_df.agent_id).union(set(payments_df.agent_id)).union(set(transactions_df.agent_id)))
    col_names = {months: f"charge_off_within_{months}_months" for months in PREDICTION_WINDOW_MONTHS}
    output_df = pd.DataFrame(columns=["agent_id"] + list(col_names.values()))
    output_df["agent_id"] = agents
    for months in PREDICTION_WINDOW_MONTHS:
        col_name = col_names[months]
        preds = np.array(
            [1]*int(co_percent * len(agents)) +
            [0]*int((1 - co_percent) * len(agents))
        )
        # When unsure of whether their predictions span the entire set of agents to predict
        #  for, the true data scientist pads their predictions with zeros lol.
        preds = np.append(preds, [0]*(len(agents) - len(preds)))
        np.random.shuffle(preds)
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