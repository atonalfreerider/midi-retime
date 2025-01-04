#!/bin/bash

# Check if we have all required arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <midi_file> <timing_file> <input_audio> <output_dir>"
    echo "Example: $0 song.mid timings.txt input.wav ./output"
    exit 1
fi

MIDI_FILE="$1"
TIMING_FILE="$2"
INPUT_AUDIO="$3"
OUTPUT_DIR="$4"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the base name of the input audio file without extension
BASENAME=$(basename "$INPUT_AUDIO" | sed 's/\.[^.]*$//')

# Create paths for temporary and output files
TEMP_JSON="$OUTPUT_DIR/${BASENAME}_timing_map.json"
OUTPUT_WAV="$OUTPUT_DIR/${BASENAME}_retimed.wav"

# Run the dictionary script to generate the JSON mapping
echo "Generating timing map..."
python retime-dictionary.py "$MIDI_FILE" "$TIMING_FILE" "$TEMP_JSON"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate timing map"
    exit 1
fi

# Run the audio stretch script using the generated JSON
echo "Stretching audio..."
python audio-stretch.py "$INPUT_AUDIO" "$TEMP_JSON" "$OUTPUT_WAV"

if [ $? -ne 0 ]; then
    echo "Error: Failed to stretch audio"
    rm -f "$TEMP_JSON"
    exit 1
fi

# Clean up
rm -f "$TEMP_JSON"

echo "Successfully created retimed audio: $OUTPUT_WAV"
