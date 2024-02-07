"""
This script is based on the jupyter notebook 'Preprocess_GMD2HVO_Sequence.ipynb'
Most of the logic is directly copied this notebook.

Retrieved from https://github.com/behzadhaki/GMD2HVO_PreProcessing
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

# Import necessary libraries for processing/loading/storing the dataset
import numpy as np
import pickle
import pandas as pd
from shutil import copy2
import json

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
from midiUtils.constants import PERC_PARTS
from midiUtils.augExamples import AugExamplesRetriever

# Import magenta's note_seq 
import note_seq

RESOURCES_DIR = 'resources'
GMD_DATASET_DIR = RESOURCES_DIR + '/gmd_dataset'
GMD_GROOVE_DIR = GMD_DATASET_DIR + '/groove'
PREPROCESSED_DATASETS_DIR = 'preprocessedDatasets'

# constants for data augmentation
SEED = 0
EXAMPLES_DIR = "examples"
TMP_DIR = 'tmp'
NUM_TRANSFORMATIONS = 1
NUM_REPLACEMENTS = 2
AER = AugExamplesRetriever(EXAMPLES_DIR)
STYLE_PARAMS = {
        "preferredStyle" : "",
        "outOfStyleProb" : 0.2
    }

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

def transform_midi(midi_data, trackIndex=1, numReplacements=1):
    """
    TODO: refactor with candidate midi
    """

    original_midi_path = f'{TMP_DIR}/original.mid'
    new_midi_path = f'{TMP_DIR}/transformed.mid'
    debug = False

    # TESTING DATA AUG LOGIC ONLY: with WRITE_PROB, select the given midi_data for evalutaion.
    # this means that we write the midi_data to file to tmp as usual but we don't overwrite it afterwards
    # this is so that we can inspect the transformation result after the run
    if TEST_DATA_AUG:
        debug = True
        global evaluation_files_counter
        if random.random() < WRITE_PROB:
            original_midi_path = f'{TMP_DIR}/test{evaluation_files_counter}.mid'
            new_midi_path = f'{TMP_DIR}/transformed_test{evaluation_files_counter}.mid'
            evaluation_files_counter += 1

    # write the midi_data to a file so that we can build a mido_file out of it.
    with open(original_midi_path, "wb") as binary_file:
        binary_file.write(midi_data)
    mido_file = mido.MidiFile(original_midi_path)

    # run random transformation 
    partsToReplace = random.sample(PERC_PARTS, numReplacements)
    new_mido_file = dataAug.transformMidiFile(mid=mido_file, trackIndex=trackIndex, partsToReplace=partsToReplace, augExamplesRetriever=AER, styleParams=STYLE_PARAMS, debug=debug)

    # save contents to file, where it will be read back into a midi_data object that we will return
    new_mido_file.save(new_midi_path)
    with open(new_midi_path, "rb") as binary_file:
        new_midi_data = binary_file.read()

    return new_midi_data

def convert_groove_midi_dataset(dataset, beat_division_factors=[4], csv_dataframe_info=None, numTransformations=0):
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
                        midi_data = features["midi"].numpy()[0] if is_original else transform_midi(features["midi"].numpy()[0], numReplacements=NUM_REPLACEMENTS)
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
                    midi_filename = df["midi_filename"].to_numpy()[0] if is_original else "None"
                    audio_filename = df["audio_filename"].to_numpy()[0] if is_original else "None"
                    
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
                        dict_append(dataset_dict_processed, "style_secondary", ["None"])
                        _hvo_seq.metadata.style_secondary = "None"

                    dict_append(dataset_dict_processed, "bpm", df["bpm"].to_numpy()[0])
                    
                    dict_append(dataset_dict_processed, "beat_type", df["beat_type"].to_numpy()[0])
                    _hvo_seq.metadata.beat_type = df["beat_type"].to_numpy()[0]
                    
                    dict_append(dataset_dict_processed, "time_signature", df["time_signature"].to_numpy()[0])
                    
                    dict_append(dataset_dict_processed, "full_midi_filename", midi_filename)
                    _hvo_seq.metadata.full_midi_filename = midi_filename

                    dict_append(dataset_dict_processed, "full_audio_filename", audio_filename)
                    _hvo_seq.metadata.full_audio_filename = audio_filename

                    dict_append(dataset_dict_processed, "midi", midi_data)
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

    json_aug = json.dumps(dataAugParams)
    with open(f'{path}/dataAugParams.json', 'w') as outfile:
        outfile.write(json_aug)

    # copy2(os.path.join(os.getcwd(), currentNotebook), path) 
    
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

def preprocess(output_dir, dataAugParams):
    dataset_train_unprocessed, dataset_train_info = tfds.load(
        name="groove/2bar-midionly",
        split=tfds.Split.TRAIN,
        try_gcs=True,
        with_info=True)

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
    print(f"unprocessed data set len: {len(list(dataset_test))}")
    dataset_validation = dataset_validation_unprocessed.batch(1)

    print("\n Number of Examples in Train Set: {}, Test Set: {}, Validation Set: {}".format(
        len(list(dataset_train)), 
        len(list(dataset_test)), 
        len(list(dataset_validation)))
        ) 
    dataframe = pd.read_csv(f"{GMD_GROOVE_DIR}/info.csv", delimiter = ',')

    numTransformations = dataAugParams["numTransformations"]

    # In groove-v1.0.0-midionly.zip, we have access to full performances, while using tfds.load(name="groove/2bar-midionly"), we can readily access the performance chopped into 2 bar segments. 
    # However, in the pre-chopped set, the meta data is missing. 
    # Hence, we need to find the relevant metadata in the info.csv file available groove-v1.0.0-midionly.zip. 
    # To do so, we match the beginning of the id in the chopped segments with the id in info.csv
    
    # Process Training Set
    dataset_train_processed = convert_groove_midi_dataset(
        dataset = dataset_train, 
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=numTransformations
    )
    print("training processing done.")
    # Process Test Set
    dataset_test_processed = convert_groove_midi_dataset(
        dataset = dataset_test, 
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=numTransformations)
    print("test processing done.")
    # Process Validation Set
    dataset_validation_processed = convert_groove_midi_dataset(
        dataset = dataset_validation, 
        beat_division_factors=[4], 
        csv_dataframe_info=dataframe,
        numTransformations=numTransformations
        )
    print("validation processing done.")

    print(f"transformation error count: {transformation_error_counter}")

    # Sort the sets using ids
    dataset_train_processed = sort_dictionary_by_key(dataset_train_processed, "loop_id")
    print("len of dataset_train_processed: ", len(dataset_train_processed["loop_id"]))
    dataset_test_processed = sort_dictionary_by_key(dataset_test_processed, "loop_id")
    print("len of dataset_test_processed: ", len(dataset_test_processed["loop_id"]))
    dataset_validation_processed = sort_dictionary_by_key(dataset_validation_processed, "loop_id")
    print("len of dataset_validation_processed: ", len(dataset_validation_processed["loop_id"]))

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
    random.seed(SEED)
    dataAugParams = {
        "seed" : SEED,
        "numTransformations" : NUM_TRANSFORMATIONS,
        "numReplacements" : NUM_REPLACEMENTS,
        "styleParams" : STYLE_PARAMS,
        "percParts" : PERC_PARTS
    }
    preprocess(PREPROCESSED_DATASETS_DIR, dataAugParams=dataAugParams)