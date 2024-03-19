from  midiUtils import synth

import os
import shutil

AUDIO_DIR = 'audio'
TMP_DIR = 'tmp'

def synthTmp():
    if os.path.exists(AUDIO_DIR):
        shutil.rmtree(AUDIO_DIR)
    os.makedirs(AUDIO_DIR)

    synth.synthesize_all(TMP_DIR, AUDIO_DIR)

if __name__ == '__main__':
    synthTmp()
    print('Done')