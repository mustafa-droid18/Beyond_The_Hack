# Beyond the Hype Hackathon Example Submission - Synthetic Credit Card Data

## Submission Requirements

The requirement for a submission is a repo with, at a minimum, two files: `__main__.py` and `env.yaml`.

The `__main__.py` file must accept two input args:
- `--bth_test_set`
    - This is a string that the grading team (not you) sets to the path of the directory in the grading compute environment that contains *test set inputs*.
    - These inputs will have the same schemas as the inputs you used for training. Use them in `__main__.py` and the files it may or may not call to run inference and log those results (see below).
- `--bth_results`
    - This is a string that the grading team (not you) sets to the path of a directory in the grading compute environment. In that directory, `__main__.py` **must** write *test set inference results* as a file named `results.csv`.
    - `results.csv` must have the same schema as the `labels.csv` you used during training.
    - For all columns but `agent_id`, `results.csv` cols must have int data types, with all values being 0 or 1. I.e. do **not** write probabilities.

The `env.yaml` file must be a valid conda environment specification with, at a minimum, the fields `name`, `channels`, and `dependencies`.

Beyond these two files, your repo can contain whatever files you like. `__main__.py` can, for
example, depend on a local model.pickle or supporting .py files. Your training/tuning scripts may 
or may not be in the repo -- our grading scripts don't care about them.

## Exploratory Data Analysis
See EDA.ipynb for an introduction to the data.