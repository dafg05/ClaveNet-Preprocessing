# ClaveNet-Preprocessing

This repository mainly houses a preprocessing module (`preprocessing.py`) for the Groove MIDI Dataset (based on [this repository](https://github.com/behzadhaki/GMD2HVO_PreProcessing)) with support for Drum Data Augmentation with Seed Patterns. Additionally, there are scripts for 1. preprocessing the ToonTrack Latin MIDI pack (`midipack_preproc.py`), 2. counting the number of examples per style in the GMD (`exampleCount`), 3. counting the number of seed patterns per style in the set of seed examples (`seedExamplesAnalysis.py`).

## Requirements

1. `$ pip install -r requirements.txt`

Note that some packages are hosted on github repos.

## Preprocessing module

The file `preprocessing.py` can be ran on its own, or as part of a batch preprocessing script (more on that later).

### Usage

```$ python -m preproc.preprocessing```

### Output

**TODO**

### Preprocessing a single-style dataset

The constant `SINGLE_STYLE` is by default set to an empty string. This setting results in preprocessing a dataset that inclues examples of all styles in the GMD.

If we set `SINGLE_STYLE = "hiphop`, the preprocessed dataset will only include examples of the "hiphop" *primary* style. Likewise for other primary styles.

### Preprocessing validation set only

To preprocess only the validation partition of the GMD, set `VALIDATION_ONLY = True`. To preprocess the train, test, and validation partitions, set `VALIDATION_ONLY = False`.

## Batch preprocessing

The script `preproc_batch.py` can be ran to preprocess multiple configurations of data augmented datasets sequentially. Each combination of the values for the number of transformations, number of replacements, and out of style probabilities results in a distinct preprocessed dataset. The output datasets will be written to the `batch_runs` directory.

### Usage

```$ python preproc_batch.py```

## Latin Midi Pack Preprocessing

**TODO**

## Counting number of examples per style in the GMD

### Prerequisite

In the file `exampleCount.py`, change the value of the constant `NON_TRANSFORMED_DATASET_ROOT` to the path of the preprocessed dataset you wish to make a count of.

### Usage

```$ python -m preproc.exampleCount```

## Counting the number of seed patterns per style in the set of seed examples

### Prerequisite 

Unzip `seedExamples.zip`. Leave its contents in the root directory. There might be `.DS_Store` files that need to be deleted. 

### Usage

```$ python -m preproc.seedExamplesAnalysis ```

