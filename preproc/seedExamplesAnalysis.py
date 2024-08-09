from pathlib import Path
from midiUtils.augExamples import SeedExamplesRetriever, AugSeedExample
from midiUtils.constants import PERC_VOICES_MAPPING

from collections import Counter
import pandas as pd
import numpy as np

base_dir = Path(__file__).parent.parent

# NOTE: Because of hardcoded paths, this script should only be run from the root directory of the project. It should be modified to do otherwise.

SEED_EXAMPLES_DIR = base_dir / "seedExamples"

SEED_EXAMPLES_32_SET = "32set"
SEED_EXAMPLES_23_SET = "23set"
SER_23 = SeedExamplesRetriever(f"{SEED_EXAMPLES_DIR}/{SEED_EXAMPLES_23_SET}")
SER_32 = SeedExamplesRetriever(f"{SEED_EXAMPLES_DIR}/{SEED_EXAMPLES_32_SET}")

def get_tracks_by_style_and_voice(ser: SeedExamplesRetriever):
    
    voices = PERC_VOICES_MAPPING.keys()
    styles = ser.styles

    df = pd.DataFrame(data=np.zeros((len(voices), len(styles))), index=voices, columns=styles)

    for style in styles:
        candidateTracks = ser.getCandidateTracksInfo(style, False, [])
        for candidateTrack in candidateTracks:
            _, voice = candidateTrack
            df.loc[voice, style] += 1

    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)

    return df

# def get_num_tracks_by_style(ser: SeedExamplesRetriever):
#     num_tracks_by_style = Counter()
    
#     for style in ser.styles:
#         numCandidateTracks = len(ser.getCandidateTracksInfo(style, False, []))
#         num_tracks_by_style[style] = numCandidateTracks

#     return num_tracks_by_style

# def get_num_tracks_by_voice(ser):
#     num_tracks_by_voice = Counter()
#     all_examples = ser.getExamplesByStyle(ser.styles[0]) + ser.getExamplesOutOfStyle(ser.styles[0])
#     for example in all_examples:
#         for voice in PERC_VOICES_MAPPING.keys():
#             if example.hasVoice(voice):
#                 num_tracks_by_voice[voice] += 1
    
#     return num_tracks_by_voice

def get_num_examples(ser):
    all_examples = ser.getExamplesByStyle(ser.styles[0]) + ser.getExamplesOutOfStyle(ser.styles[0])
    return len(all_examples)

if __name__ == "__main__":
    num_examples_23 = get_num_examples(SER_23)
    num_examples_32 = get_num_examples(SER_32)

    track_df_23 = get_tracks_by_style_and_voice(SER_23)
    track_df_32 = get_tracks_by_style_and_voice(SER_32)

    track_df_23.to_csv(SEED_EXAMPLES_DIR / f"{SEED_EXAMPLES_23_SET}_tracks_example_count_{num_examples_23}_test.csv")
    track_df_32.to_csv(SEED_EXAMPLES_DIR / f"{SEED_EXAMPLES_32_SET}_tracks_example_count_{num_examples_32}_test.csv")
