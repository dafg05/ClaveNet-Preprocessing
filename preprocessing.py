"""
This script is based on the jupyter notebook 'Preprocess_GMD2HVO_Sequence.ipynb'
Most of the logic is directly copied this notebook, with the exception of the seeded data augmentation logic

Retrieved from https://github.com/behzadhaki/GMD2HVO_PreProcessing
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

# Import necessary libraries for processing/loading/storing the dataset
import numpy as np
import pickle
import pandas as pd
import shutil
from shutil import copy2
import json
from pathlib import Path

# Import libraries for creating/naming folders/files
import os, sys
from datetime import datetime

# Import the HVO_Sequence implementation
from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

# Import libraries for data aug
import mido
import random
from midiUtils import dataAug
from midiUtils.augExamples import SeedExamplesRetriever

# Import magenta's note_seq 
import note_seq

RESOURCES_DIR = 'resources'
INFO_CSV = RESOURCES_DIR + '/info.csv'
PREPROCESSED_DATASETS_DIR = 'preprocessedDatasets'

# constants for data augmentation
NO_VALUE_STR = "N/A"
SEED_EXAMPLES_DIR = "seedExamples"
TMP_DIR = 'tmp'
TRACK_INDEX = 1
CHANNEL = 9
SEED_EXAMPLES_32_SET = "32set"
SEED_EXAMPLES_23_SET = "23set"
SER_23 = SeedExamplesRetriever(f"{SEED_EXAMPLES_DIR}/{SEED_EXAMPLES_23_SET}")
SER_32 = SeedExamplesRetriever(f"{SEED_EXAMPLES_DIR}/{SEED_EXAMPLES_32_SET}")

SERS = [SER_23, SER_32]

# tranformation parameters
RANDOM_SEED = 0
RNG = np.random.default_rng(seed = RANDOM_SEED)
NUM_TRANSFORMATIONS = 1
NUM_REPLACEMENTS = 2
OUT_OF_STYLE_PROB = 0.2

# testing data aug
TEST_DATA_AUG = False
evaluation_files_counter = 0
WRITE_PROB = 0.003

# keeping track of transformation errors:
transformation_error_counter = 0

def dict_append(dictionary, key, vals):
    """
    Appends a value or a list of values to a key in a dictionary
    """
    
    # if the values for a key are not a list, they are converted to a list and then extended with vals
    dictionary[key]=list(dictionary[key]) if not isinstance(dictionary[key], list) else dictionary[key]
    # if vals is a single value (not a list), it's converted to a list so as to be iterable
    vals = [vals] if not isinstance(vals, list) else vals
    # append new values 
    for val in vals:
        dictionary[key].append(val)

    return dictionary

def format_tuples(tuples_list):
    """
    Convert each tuple in the list to a string where its elements are joined by ':'
    Then join these strings with ';' to form the final string
    """
    formatted_string = ';'.join([":".join(map(str, tup)) for tup in tuples_list])
    return formatted_string

def transform_midi(midi_data, dataAugParams):
    """
    TODO: Use arguments instead of using constants.
    """
    original_midi_path = f'{TMP_DIR}/original.mid'
    new_midi_path = f'{TMP_DIR}/transformed.mid'
    debug = False

    # choose SERs randomly. Choose a random style from the chosen SER.
    ser = RNG.choice(SERS)
    preferredStyle = RNG.choice(ser.styles)

    # TESTING DATA AUG LOGIC ONLY: with WRITE_PROB, select the given midi_data for evalutaion.
    # this means that we write the midi_data to file to tmp as usual but we don't overwrite it afterwards
    # this is so that we can inspect the transformation result after the run
    if TEST_DATA_AUG:
        global evaluation_files_counter
        if random.random() < WRITE_PROB:
            print(f"Writing midi data to file for evaluation. Iteration {evaluation_files_counter}. PreferredStyle: {preferredStyle}, SER dir: {ser.dir}")
            original_midi_path = f'{TMP_DIR}/test{evaluation_files_counter}.mid'
            new_midi_path = f'{TMP_DIR}/transformed_test{evaluation_files_counter}.mid'
            debug = True
            evaluation_files_counter += 1

    # write the midi_data to a file so that we can build a mido_file out of it.
    with open(original_midi_path, "wb") as binary_file:
        binary_file.write(midi_data)
    mido_file = mido.MidiFile(original_midi_path)

    numReplacements = dataAugParams["numReplacements"]
    outOfStyleProb = dataAugParams["outOfStyleProb"]

    # run random transformation, preserve the replacement tracks information
    new_mido_file, replacementInfo = dataAug.transformMidiFile(mido_file, trackIndex=TRACK_INDEX, numReplacements=numReplacements, ser=ser, rng=RNG, preferredStyle=preferredStyle, outOfStyleProb=outOfStyleProb, debug=debug)

    # save contents to file, where it will be read back into a midi_data object that we will return
    new_mido_file.save(new_midi_path)
    with open(new_midi_path, "rb") as binary_file:
        new_midi_data = binary_file.read()

    return new_midi_data, preferredStyle, replacementInfo

def convert_groove_midi_dataset(dataset, dataAugParams, beat_division_factors=[4], csv_dataframe_info=None, numTransformations=0):
    """
    Converts a tfds dataset into a dictionary containing the processed HVO_Sequence objects and metadata.
    """ 
    dataset_dict_processed = dict()
    dataset_dict_processed.update({
        "drummer":[],
        "session":[],
        "loop_id":[],  # the id of the recording from which the loop is extracted
        "master_id":[], # the id of the recording from which the loop is extracted
        "style_primary":[],
        "style_secondary":[],
        "bpm":[],
        "beat_type":[],
        "time_signature":[],
        "full_midi_filename":[],
        "full_audio_filename":[],
        "midi":[],
        "preferredStyle":[], # this is a new key that we add to the dictionary to keep track of the preferred style for the transformation
        "replacementTracksInfo":[], # this is a new key that we add to the dictionary to keep track of the replacement tracks for the transformation
        "note_sequence":[],
        "hvo_sequence":[],
    })
    for features in dataset:
        # Features to be extracted from the dataset
        note_sequence = note_seq.midi_to_note_sequence(tfds.as_numpy(features["midi"][0]))
        
        if note_sequence.notes: # ignore if no notes in note_sequence (i.e. empty 2 bar sequence) 
            _hvo_seq = note_sequence_to_hvo_sequence(
                ns = note_sequence, 
                drum_mapping = ROLAND_REDUCED_MAPPING,
                beat_division_factors = beat_division_factors
            )
            
            if (not csv_dataframe_info.empty) and len(_hvo_seq.time_signatures)==1 and len(_hvo_seq.tempos)==1 :
                for i in range(numTransformations + 1):
                    # Transformed entries in the dictionary differ from their original counterparts in the following ways:
                    # - midi, note_seq, and hvo_sequence reflect the transformation of the midi data
                    # - loop_id will be prefixed by 'transformed_{transformation_number}/'
                    # - full_midi_filename and full_audio_name will be replaced with 'None'
                    # The rest of the data is identical to the original entry.

                    # Master ID for the Loop
                    main_id = features["id"].numpy()[0].decode("utf-8").split(":")[0]

                    # Get the relevant series from the dataframe
                    df = csv_dataframe_info[csv_dataframe_info.id == main_id]
                    
                    # if we're working with a transformation, update data accordingly
                    is_original = i == 0
                    try:
                        # midi_data = features["midi"].numpy()[0] if is_original else transform_midi(features["midi"].numpy()[0])
                        if is_original:
                            midi_data, preferredStyle, replacementInfo = features["midi"].numpy()[0], NO_VALUE_STR, NO_VALUE_STR
                        else:
                            midi_data, preferredStyle, replacementInfo = transform_midi(features["midi"].numpy()[0], dataAugParams)
                            replacementInfo = format_tuples(replacementInfo)
                        note_sequence = note_sequence if is_original else note_seq.midi_to_note_sequence(midi_data)
                        # there's a chance that a transformation could result in an empty midi sequence. 
                        # if this is the case, we simply skip the transformation
                        if not note_sequence.notes:
                            continue
                        _hvo_seq = _hvo_seq if is_original else note_sequence_to_hvo_sequence(
                            ns = note_sequence, 
                            drum_mapping = ROLAND_REDUCED_MAPPING,
                            beat_division_factors = beat_division_factors
                        )
                    except Exception as e:
                        if is_original:
                            raise e
                        else:
                            print(f"Error transforming midi file: {e}")
                            global transformation_error_counter
                            transformation_error_counter += 1
                            continue
                    loop_id = features["id"].numpy()[0].decode("utf-8") if is_original else f"transformed_{i:02d}/{loop_id}"
                    midi_filename = df["midi_filename"].to_numpy()[0] if is_original else NO_VALUE_STR
                    audio_filename = df["audio_filename"].to_numpy()[0] if is_original else NO_VALUE_STR
                    
                    # Update the dictionary associated with the metadata
                    dict_append(dataset_dict_processed, "drummer", df["drummer"].to_numpy()[0])
                    _hvo_seq.metadata.drummer = df["drummer"].to_numpy()[0]
                    
                    dict_append(dataset_dict_processed, "session", df["session"].to_numpy()[0].split("/")[-1])
                    _hvo_seq.metadata.session = df["session"].to_numpy()[0]
                    
                    # !! Transformation change!
                    dict_append(dataset_dict_processed, "loop_id", loop_id)
                    _hvo_seq.metadata.loop_id = loop_id

                    dict_append(dataset_dict_processed, "master_id", main_id)
                    _hvo_seq.metadata.master_id = main_id

                    style_full = df["style"].to_numpy()[0]
                    style_primary = style_full.split("/")[0]
                    
                    dict_append(dataset_dict_processed, "style_primary", style_primary)
                    _hvo_seq.metadata.style_primary = style_primary
                    
                    if "/" in style_full:
                        style_secondary = style_full.split("/")[1]
                        dict_append(dataset_dict_processed, "style_secondary", style_secondary)
                        _hvo_seq.metadata.style_secondary = style_secondary
                    else:
                        dict_append(dataset_dict_processed, "style_secondary", [NO_VALUE_STR])
                        _hvo_seq.metadata.style_secondary = NO_VALUE_STR

                    dict_append(dataset_dict_processed, "bpm", df["bpm"].to_numpy()[0])
                    
                    dict_append(dataset_dict_processed, "beat_type", df["beat_type"].to_numpy()[0])
                    _hvo_seq.metadata.beat_type = df["beat_type"].to_numpy()[0]
                    
                    dict_append(dataset_dict_processed, "time_signature", df["time_signature"].to_numpy()[0])
                    
                    dict_append(dataset_dict_processed, "full_midi_filename", midi_filename)
                    _hvo_seq.metadata.full_midi_filename = midi_filename

                    dict_append(dataset_dict_processed, "full_audio_filename", audio_filename)
                    _hvo_seq.metadata.full_audio_filename = audio_filename

                    dict_append(dataset_dict_processed, "midi", midi_data)

                    dict_append(dataset_dict_processed, "preferredStyle", preferredStyle)

                    dict_append(dataset_dict_processed, "replacementTracksInfo", replacementInfo)

                    dict_append(dataset_dict_processed, "note_sequence", [note_sequence])
                            
                    dict_append(dataset_dict_processed, "hvo_sequence", _hvo_seq)
        else:
            pass 
    return dataset_dict_processed

def sort_dictionary_by_key (dictionary_to_sort, key_used_to_sort):
    # sorts a dictionary according to the list within a given key
    sorted_ids=np.argsort(dictionary_to_sort[key_used_to_sort])
    for key in dictionary_to_sort.keys():
        dictionary_to_sort[key]=[dictionary_to_sort[key][i] for i in sorted_ids]
    return dictionary_to_sort

# DUMP INTO A PICKLE FILE
def store_dataset_as_pickle(dataset_list, 
                            filename_list, 
                            dataAugParams,
                            root_dir = "processed_dataset",
                            append_datetime=True, 
                            features_with_separate_picklefile = ["hvo", "midi", "note_seq"]
                           ):

    #filename = filename.split(".obj")[0]
    path = root_dir
    
    if append_datetime:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_at_%H_%M_hrs")
    else:
        dt_string =""
        
    path = os.path.join(path, "PreProcessed_On_"+dt_string)
    
    if not os.path.exists (path):
        os.makedirs(path)

    # Save the dataAugParams
    json_aug = json.dumps(dataAugParams, indent = 4)
    with open(f'{path}/dataAugParams.json', 'w') as outfile:
        outfile.write(json_aug)

    
    for i, dataset in enumerate(dataset_list):

        subdirectory = os.path.join(path, filename_list[i])
        if not os.path.exists (subdirectory):
            os.makedirs(subdirectory)
            
        print("-"*100)
        print("-"*100)
        print("Processing %s folder" % subdirectory)
        print("-"*100)
        print("-"*100)
        
        # Create Metadata File
        csv_dataframe = pd.DataFrame()
        
        for k in dataset.keys():
            if k not in features_with_separate_picklefile:
                csv_dataframe[k] = dataset[k]
        
        csv_dataframe.to_csv(os.path.join(subdirectory, "metadata.csv"))
        
        print("Metadata created!")
        print("-"*100)

        for feature in features_with_separate_picklefile:
            if feature in dataset.keys():
                dataset_filehandler = open(os.path.join(subdirectory, "%s_data.obj"%feature),"wb")
                print(feature)
                print(dataset_filehandler)
                pickle.dump(dataset[feature],  dataset_filehandler)
                dataset_filehandler.close()
                print("feature %s pickled at %s" % (feature, os.path.join(subdirectory, "%s.obj"%filename_list[i].split(".")[0])))
                print("-"*100)

            else:
                 raise Warning("Feature is not available: ", feature)

def preprocess_validation_only(output_dir, dataAugParams):
    """
    Used for testing purposes. Preprocesses only the validation set.
    """
    dataset_validation_unprocessed = tfds.load(
        name="groove/2bar-midionly",
        split=tfds.Split.VALIDATION,
        try_gcs=True)
    
    dataset_validation = dataset_validation_unprocessed.batch(1)

    dataframe = pd.read_csv(INFO_CSV, delimiter = ',')

    numTransformations = dataAugParams["numTransformations"]

    dataset_validation_processed = convert_groove_midi_dataset(
        dataset = dataset_validation, 
        dataAugParams=dataAugParams,
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=numTransformations
        )
    print("Validation processing done.")

    print(f"Transformation error count: {transformation_error_counter}")

    dataset_validation_processed = sort_dictionary_by_key(dataset_validation_processed, "loop_id")
    print("Len of dataset_validation_processed: ", len(dataset_validation_processed["loop_id"]))

    dataset_list = [dataset_validation_processed]

    filename_list = ["GrooveMIDI_processed_validation"]

    store_dataset_as_pickle(dataset_list, 
                            filename_list,
                            dataAugParams=dataAugParams,
                            root_dir=output_dir,
                            append_datetime=True,
                            features_with_separate_picklefile = ["hvo_sequence", "midi", "note_sequence"]
                       )

def preprocess(output_dir, dataAugParams):
    """
    Preprocesses the GMD dataset and stores the training, test, and validation sets as pickles.
    Only applies data augmentation to the training set.
    """

    dataset_train_unprocessed = tfds.load(
        name="groove/2bar-midionly",
        split=tfds.Split.TRAIN,
        try_gcs=True)

    dataset_test_unprocessed = tfds.load(
        name="groove/2bar-midionly",
        split=tfds.Split.TEST,
        try_gcs=True)

    dataset_validation_unprocessed = tfds.load(
        name="groove/2bar-midionly",
        split=tfds.Split.VALIDATION,
        try_gcs=True)
    
    # In all three sets, separate entries into individual examples 
    dataset_train = dataset_train_unprocessed.batch(1)
    dataset_test  = dataset_test_unprocessed.batch(1)
    dataset_validation = dataset_validation_unprocessed.batch(1)

    print("\n Number of Examples in Train Set: {}, Test Set: {}, Validation Set: {}".format(
        len(list(dataset_train)), 
        len(list(dataset_test)), 
        len(list(dataset_validation)))
        ) 
    dataframe = pd.read_csv(INFO_CSV, delimiter = ',')

    numTransformations = dataAugParams["numTransformations"]

    # In groove-v1.0.0-midionly.zip, we have access to full performances, while using tfds.load(name="groove/2bar-midionly"), we can readily access the performance chopped into 2 bar segments. 
    # However, in the pre-chopped set, the meta data is missing. 
    # Hence, we need to find the relevant metadata in the info.csv file available groove-v1.0.0-midionly.zip. 
    # To do so, we match the beginning of the id in the chopped segments with the id in info.csv
    
    # Process Training Set. Augmentation is applied here
    dataset_train_processed = convert_groove_midi_dataset(
        dataset = dataset_train, 
        dataAugParams=dataAugParams,
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=numTransformations
    )
    print("Training processing done.")
    # Process Test Set. Skip augmentation here
    dataset_test_processed = convert_groove_midi_dataset(
        dataset = dataset_test, 
        dataAugParams=dataAugParams,
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=0)
    print("Test processing done.")
    # Process Validation Set. Skip augmentation here
    dataset_validation_processed = convert_groove_midi_dataset(
        dataset = dataset_validation, 
        dataAugParams=dataAugParams,
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=0
        )
    print("Validation processing done.")

    print(f"Transformation error count: {transformation_error_counter}")

    # Sort the sets using ids
    dataset_train_processed = sort_dictionary_by_key(dataset_train_processed, "loop_id")
    print("Len of dataset_train_processed: ", len(dataset_train_processed["loop_id"]))
    dataset_test_processed = sort_dictionary_by_key(dataset_test_processed, "loop_id")
    print("Len of dataset_test_processed: ", len(dataset_test_processed["loop_id"]))
    dataset_validation_processed = sort_dictionary_by_key(dataset_validation_processed, "loop_id")
    print("Len of dataset_validation_processed: ", len(dataset_validation_processed["loop_id"]))

    dataset_list = [dataset_train_processed,
               dataset_test_processed,
               dataset_validation_processed]

    filename_list = ["GrooveMIDI_processed_train",
                    "GrooveMIDI_processed_test",
                    "GrooveMIDI_processed_validation"]

    store_dataset_as_pickle(dataset_list, 
                            filename_list,
                            dataAugParams=dataAugParams,
                            root_dir=output_dir,
                            append_datetime=True,
                            features_with_separate_picklefile = ["hvo_sequence", "midi", "note_sequence"]
                       )

if __name__ == "__main__":
    random.seed(RANDOM_SEED)

    dataAugParams = {
        "seed" : RANDOM_SEED,
        "seedExamplesSets" : [Path(SER_23.dir).name, Path(SER_32.dir).name],
        "numTransformations" : NUM_TRANSFORMATIONS,
        "numReplacements" : NUM_REPLACEMENTS,
        "outOfStyleProb" : OUT_OF_STYLE_PROB
    }

    # delete the tmp folder if it exists
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    # create a new tmp folder
    os.makedirs(TMP_DIR)

    preprocess(PREPROCESSED_DATASETS_DIR, dataAugParams=dataAugParams)
    # preprocess_validation_only(PREPROCESSED_DATASETS_DIR, dataAugParams=dataAugParams)