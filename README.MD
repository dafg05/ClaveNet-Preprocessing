# ClaveNet-Preprocessing

This repository mainly houses a preprocessing module (`preprocessing.py`) for the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) (based on [this repository](https://github.com/behzadhaki/GMD2HVO_PreProcessing)) with support for Drum Data Augmentation with Seed Patterns. Additionally, there are scripts for 1. preprocessing the ToonTrack Latin MIDI pack (`midipack_preproc.py`), 2. counting the number of examples per style in the GMD (`exampleCount`), 3. counting the number of seed patterns per style and drum voice in the set of seed examples (`seedExamplesAnalysis.py`).

## Requirements

1. `$ pip install -r requirements.txt`

Note that some packages are hosted on github repos.

## Preprocessing module

The file `preprocessing.py` can be ran on its own or as part of a batch preprocessing script (see Batch preprocessing section).

### Usage

```$ python -m preproc.preprocessing```

### Output

The preprocessed dataset is a directory written to `out/` structured as follows:

```
PreProcessed_On_DD_MM_YYYY_at_HH_MM_hrs/
|-- GrooveMIDI_processed_test/
    |-- hvo_sequence_data.obj
    |-- metadata.csv
    |-- midi_data.obj
    |-- note_sequence_data.obj
|-- GrooveMIDI_processed_train/
    |-- ...
|-- GrooveMIDI_processed_validation/
    |-- ...
|-- dataAugParams.json
```

The file `dataAugParams.json` details the data augmentation parameter configuration used to generate the preprocessed dataset. The pickled `.obj` files contain different symbolic representations for the GMD's drum recordings. They share indices; e.g. the 50th HVO Sequence in `hvo_seq.obj` represents the same drum recording as the 50th MIDI file in `midi_data.obj`.

### Data augmentation parameters

To customize data augmentation parameters, change the values of the constants `NUM_TRANSFORMATIONS`, `NUM_REPLACEMENTS`, `OUT_OF_STYLE_PROB`. To preprocess the vanilla GMD, simply set `NUM_TRANSFORMATIONS = 0`.

### Preprocessing options

* The constant `SINGLE_STYLE` is by default set to an empty string. This setting results in preprocessing a dataset that inclues examples of all styles in the GMD. If we set `SINGLE_STYLE = "hiphop`, the preprocessed dataset will only include examples of the "hiphop" *primary* style. Likewise for other primary styles.
* To preprocess only the validation partition of the GMD, set `VALIDATION_ONLY = True`. To preprocess the train, test, and validation partitions, set `VALIDATION_ONLY = False`.

### Batch preprocessing

The script `preproc_batch.py` can be ran to preprocess multiple configurations of data augmented datasets sequentially. Each combination of the values for the number of transformations, number of replacements, and out of style probabilities results in a distinct preprocessed dataset. 

#### Usage

```$ python preproc_batch.py```

The output preprocessed datasets will be written to the `batch_runs` directory.

## Latin Midi Pack Preprocessing

**TODO**

## Counting number of examples per style in the GMD

### Usage
1. In the file `exampleCount.py`, change the value of the constant `NON_TRANSFORMED_DATASET_ROOT` to the path of the preprocessed dataset you wish to make a count of.
2. ```$ python -m preproc.exampleCount```

The per-style count will be printed to the console.

## Counting the number of seed patterns per style and drum voice in the set of seed examples

### Usage
1. Unzip `seedExamples.zip`. Leave its contents in the root directory. There might be `.DS_Store` files that need to be deleted. 
2. ```$ python -m preproc.seedExamplesAnalysis ```

Two `.csv` files --one for the 23 set and the other for the 32 set-- will be written to the `seedExamples` dir.
