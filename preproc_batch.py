import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from preproc import preprocessing as pre

PREPROCESSING_RUNS_DIR = Path('batch_runs')
TRAINING_ERROR_LOGS = 'preprocessing_errors.log'

NUM_TRANSFORMATION_VALUES = [1]
NUM_REPLACEMENT_VALUES = [1, 2]
OUT_OF_STYLE_PROB_VALUES = [0.5]

def pre_processing_pipeline():
    # Preprocess dataset with different data augmentation parameters.

    # name each run with a timestamp
    time_str = str(int(datetime.now().timestamp()))
    run_path = Path(PREPROCESSING_RUNS_DIR, time_str)
    Path.mkdir(run_path, exist_ok=True)

    error_log_path = Path(run_path, TRAINING_ERROR_LOGS)
    with open(error_log_path, 'w') as f:
        f.write(f"Preprocessing error log for run {time_str} \n")

    combinations = get_data_aug_params_combinations(NUM_TRANSFORMATION_VALUES, NUM_REPLACEMENT_VALUES, OUT_OF_STYLE_PROB_VALUES)

    error_count = 0
    for data_aug_params in tqdm(combinations, desc="Processing pipeline"):
        try:
            # preprocess the dataset
            pre.preprocess(run_path, data_aug_params)
        
        except Exception as e:
            print("An error occured while preprocessing the dataset.")
            with open(error_log_path, 'a') as f:
                f.write(f"Error preprocessing with data aug params: {data_aug_params}. {traceback.format_exc()} \n")
                error_count += 1

    print(f"Preprocessed {len(combinations) - error_count} out of {len(combinations)} combinations. Errors written on {error_log_path}. Check {run_path} for processed datasets.")


def get_data_aug_params_combinations(num_transformation_values, num_replacement_values, out_of_style_prob_values):
    combinations = []
    for t in num_transformation_values:
        for r in num_replacement_values:
            for o in out_of_style_prob_values:
                data_aug_params = {
                    "random_seed" : pre.RANDOM_SEED,
                    "seed_examples_sets" : pre.SEED_EXAMPLES_SETS,
                    "num_transformations" : t,
                    "num_replacements" : r,
                    "out_of_style_prob" : o
                }
                combinations.append(data_aug_params)
    return combinations


if __name__ == "__main__":
    pre_processing_pipeline()
        