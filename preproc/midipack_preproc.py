from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle

from preproc.preprocessing import dict_append, sort_dictionary_by_key

import note_seq
from hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

base_dir = Path(__file__).parent.parent

# NOTE: Because of hardcoded paths, this script should only be run from the root directory of the project. It should be modified to do otherwise.

MIDIPACK_DIR = base_dir / 'toontrack_latin_midi'
PREPROCESSED_DATASETS_DIR = base_dir / 'preprocessedDatasets'

MODIFIED_MAPPING = ROLAND_REDUCED_MAPPING.copy()
MODIFIED_MAPPING["HH_CLOSED"].append(71) # Looking at the guaguanco midi for toontrack midi pack, there's a palito pattern on note 71. So, adding it to the hi-hat mapping

MIDIPACK_NAME = "Toontrack"
DRUMMER = "mauricio_herrera"
SESSION = "toontrack_latin_rhythms_midi"
STYLE_PRIMARY = "latin"
TIME_SIGNATURE = "4-4"
BEAT_TYPE = "beat"

def store_midipack_dataset_as_pickle(dataset, 
                            root_dir:Path,
                            append_datetime=True, 
                            features_with_separate_picklefile = ["hvo_sequence", "midi", "note_sequence"]
                           ):
    if append_datetime:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_at_%H_%M_hrs")
    else:
        dt_string =""
        
    out_dir = Path(root_dir, f"{MIDIPACK_NAME}_PreProcessed_On_{dt_string}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Metadata File
    csv_dataframe = pd.DataFrame()
    
    for k in dataset.keys():
        if k not in features_with_separate_picklefile:
            csv_dataframe[k] = dataset[k]
    
    csv_dataframe.to_csv(Path(out_dir, "metadata.csv"))
    
    print("Metadata created!")
    print("-"*100)

    # Pickle features not in the metadata
    for feature in features_with_separate_picklefile:
        if feature in dataset.keys():
            with open(Path(out_dir, f"{feature}_data.obj"), "wb") as feature_file:
                print(feature)
                pickle.dump(dataset[feature], feature_file)
                print(f"Feature pickled at {out_dir}/{feature}_data.obj")
                print("-"*100)

        else:
            raise Warning("Feature is not available: ", feature)
        
def create_midipack_dataset(midipack_dir):
    dataset_dict_processed = dict()
    dataset_dict_processed.update({
        "drummer":[],
        "session":[],
        "master_id":[], # the id of the recording from which the loop is extracted
        "style_primary":[],
        "style_secondary":[],
        "beat_type":[],
        "time_signature":[],
        "full_midi_filename":[],
        "midi":[],
        "note_sequence":[],
        "hvo_sequence":[],
    })

    drummer = DRUMMER
    session = SESSION
    style_primary = STYLE_PRIMARY
    time_signature = TIME_SIGNATURE
    beat_type = BEAT_TYPE

    for midi_path in midipack_dir.glob("*.mid"):
        with open(midi_path, "rb") as midi_file:
            midi_data = midi_file.read()
            note_sequence = note_seq.midi_to_note_sequence(midi_data)
            if note_sequence.notes:
                hvo_sequence = note_sequence_to_hvo_sequence(ns=note_sequence, drum_mapping=MODIFIED_MAPPING,beat_division_factors=[4])

                dict_append(dataset_dict_processed, "drummer", drummer)
                dict_append(dataset_dict_processed, "session", session)
                dict_append(dataset_dict_processed, "style_primary", style_primary)
                dict_append(dataset_dict_processed, "style_secondary", midi_path.name.split("_")[0])
                dict_append(dataset_dict_processed, "time_signature", time_signature)
                dict_append(dataset_dict_processed, "beat_type", beat_type)
                dict_append(dataset_dict_processed, "master_id", midi_path.stem)
                dict_append(dataset_dict_processed, "full_midi_filename", f"{midi_path.parent.name}/{midi_path.name}")
                dict_append(dataset_dict_processed, "midi", midi_data)
                dict_append(dataset_dict_processed, "note_sequence", note_sequence)
                dict_append(dataset_dict_processed, "hvo_sequence", hvo_sequence)

    return dataset_dict_processed

def preprocess_midipack_dataset(midipack_dir, output_dir):
    print("Preprocessing Midipack Dataset")
    dataset = create_midipack_dataset(midipack_dir)
    dataset = sort_dictionary_by_key(dataset, "master_id")
    print("Dataset created!")
    print("-"*100)
    
    store_midipack_dataset_as_pickle(dataset, output_dir)
    print("Dataset stored!")

if __name__ == "__main__":
    preprocess_midipack_dataset(MIDIPACK_DIR, PREPROCESSED_DATASETS_DIR)
