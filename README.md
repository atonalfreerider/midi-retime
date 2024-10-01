# midi-retime

The purpose of this tool is to retime a MIDI file to match the tempo of another MIDI file.

Master mid vs Input wav:
|-----|-----|----------|---|
|---|-----|-------|---|


Master mid vs retimed wav:
|-----|-----|----------|---|
|-----|-----|----------|---|

The midi file that will be retimed, may not necessarily have the same note structure as the file that it is being retimed to. The source of the midi file might be from a MIDI recorder, or from a MIDI editor. The MIDI file that it is being retimed to, might be a MIDI file that is intended to be the "master", that other MIDI files are trying to match. 

The tool is flexible in that it can retime a MIDI file to match the tempo of another MIDI file, even if the two MIDI files have different note structure. 

retetime-dictionary.py takes a .txt file dictionary that indicates the start time of each measure in the audio file to be retimed, and outputs a corse retime scale.
midi-note-analysis.py further takes the retime scale and subdivides the midi notes so that it can maximize overlap and minimize non-overlap, and produces a fine grain retime scale.
audio-stretch.py takes the retime scale and retimes the audio file to match the tempo of the master midi file.

Installation:

pip install -r requirements.txt

sudo apt-get update
sudo apt-get install libsndfile1-dev librubberband-dev

git clone https://github.com/bmcfee/pyrubberband.git
cd pyrubberband
pip install -e .

sudo apt-get update
sudo apt-get install rubberband-cli



Convert audio to midi using:
https://github.com/spotify/basic-pitch



