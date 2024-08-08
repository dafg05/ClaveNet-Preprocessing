from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

# strings
TRAIN_SUFFIX = 'train'
TEST_SUFFIX = 'test'
VALIDATION_SUFFIX = 'validation'
PARTITION_PREFIX = 'GrooveMIDI_processed_'
METADATA_FILE = 'metadata.csv'

# paths
PREPROCESSED_DATASETS = Path('preprocessedDatasets')

# preprocessed dataset path: change this to the path of the dataset you want to analyze
NON_TRANSFORMED_DATASET_ROOT = PREPROCESSED_DATASETS / 'non_transformed_datasets'

TRAIN_PARTITION = NON_TRANSFORMED_DATASET_ROOT / f'{PARTITION_PREFIX}{TRAIN_SUFFIX}'
TEST_PARTITION = NON_TRANSFORMED_DATASET_ROOT / f'{PARTITION_PREFIX}{TEST_SUFFIX}'
VALIDATION_PARTITION = NON_TRANSFORMED_DATASET_ROOT / f'{PARTITION_PREFIX}{VALIDATION_SUFFIX}'


# functions
def count_exercises_by_style(partition):
    metadata = pd.read_csv(partition / METADATA_FILE)
    return metadata.groupby('style_primary').size()

def convert_count_to_percentage(count):
    return count / count.sum()


if __name__ == '__main__':
    train_count = count_exercises_by_style(TRAIN_PARTITION)
    test_count = count_exercises_by_style(TEST_PARTITION)
    validation_count = count_exercises_by_style(VALIDATION_PARTITION)

    combined_count = train_count.add(test_count, fill_value=0)
    combined_count = combined_count.add(validation_count, fill_value=0)

    other_total = combined_count[combined_count < 750].sum()
    other_data = {'other': other_total}
    other_count = pd.Series(other_data, index=['other'])

    new_combined_count = combined_count.where(combined_count >= 750)
    new_combined_count.dropna(inplace=True)
    new_combined_count = new_combined_count.add(other_count, fill_value=0).sort_values(ascending=False)

    percentage_count = convert_count_to_percentage(new_combined_count)

    assert combined_count.sum() == new_combined_count.sum()
    assert np.isclose(percentage_count.sum(), 1.0, atol=1e-5)

    print('Combined count:')
    print(combined_count)
    print(f'\nTotal: {combined_count.sum()}')
    print('New combined count:')
    print(new_combined_count)
    print(f'\nTotal: {new_combined_count.sum()}')
    print('Percentage:')
    print(convert_count_to_percentage(new_combined_count))


