import argparse
import numpy as np
import librosa
import soundfile as sf
import mido
from typing import Dict, Tuple

def parse_timing_dict(timing_file: str) -> Dict[int, float]:
    timing_dict = {}
    with open(timing_file, 'r') as f:
        for line in f:
            measure, time_str = line.strip().split('|')
            measure = int(measure)
            minutes, seconds = time_str.split(':')
            seconds, milliseconds = seconds.split('.')
            time_in_seconds = (int(minutes) * 60) + int(seconds) + (int(milliseconds) / 1000)
            timing_dict[measure] = time_in_seconds
    return timing_dict

def get_midi_timings(midi_file: str) -> Dict[int, float]:
    midi_timings = {}
    midi = mido.MidiFile(midi_file)
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # Default tempo (microseconds per beat)
    time_signature = (4, 4)  # Default time signature
    total_ticks = 0
    current_measure = 1
    ticks_per_measure = ticks_per_beat * 4  # Default 4/4 time signature

    # Get the total duration of the MIDI file in seconds
    total_duration_seconds = midi.length

    # Combine all tracks into a single list of messages, sorted by their absolute time
    all_messages = []
    for track in midi.tracks:
        track_ticks = 0
        for msg in track:
            track_ticks += msg.time
            all_messages.append((track_ticks, msg))
    all_messages.sort(key=lambda x: x[0])

    # Find the total duration of the MIDI file in ticks
    total_duration_ticks = max(msg[0] for msg in all_messages)

    for total_ticks, msg in all_messages:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'time_signature':
            time_signature = (msg.numerator, msg.denominator)
            ticks_per_measure = ticks_per_beat * 4 * time_signature[0] // time_signature[1]

        # Calculate the current measure
        current_measure = 1 + total_ticks // ticks_per_measure

        # Convert ticks to seconds
        seconds = (total_ticks / total_duration_ticks) * total_duration_seconds

        # Update midi_timings if this is a new measure or later timing for the same measure
        if current_measure not in midi_timings or seconds > midi_timings[current_measure]:
            midi_timings[current_measure] = seconds

    # Ensure we capture the last measure and the total duration
    last_measure = 1 + total_duration_ticks // ticks_per_measure
    midi_timings[last_measure] = total_duration_seconds

    return midi_timings

def create_stretching_map(wav_timings: Dict[int, float], midi_timings: Dict[int, float]) -> Dict[float, float]:
    stretching_map = {}
    wav_measures = sorted(wav_timings.keys())
    midi_measures = sorted(midi_timings.keys())
    
    for i in range(len(wav_measures) - 1):
        start_measure = wav_measures[i]
        end_measure = wav_measures[i + 1]
        
        wav_start = wav_timings[start_measure]
        wav_end = wav_timings[end_measure]
        
        # Find the corresponding MIDI measures
        midi_start_measure = next((m for m in midi_measures if m >= start_measure), None)
        midi_end_measure = next((m for m in midi_measures if m >= end_measure), None)
        
        if midi_start_measure is None or midi_end_measure is None:
            print(f"Warning: Could not find corresponding MIDI measures for WAV measures {start_measure} to {end_measure}")
            continue
        
        midi_start = midi_timings[midi_start_measure]
        midi_end = midi_timings[midi_end_measure]
        
        stretch_factor = (midi_end - midi_start) / (wav_end - wav_start)
        stretching_map[wav_start] = (midi_start, stretch_factor)
    
    return stretching_map

def retime_audio(input_wav: str, stretching_map: Dict[float, Tuple[float, float]], output_wav: str):
    # Load the audio file
    y, sr = librosa.load(input_wav, sr=None)
    
    # Create time array
    time = np.arange(len(y)) / sr
    
    # Sort the stretching map by original time
    sorted_stretch_points = sorted(stretching_map.items())
    
    # Initialize arrays for the new time points
    new_times = []
    original_times = []
    
    # Process each stretch point
    for i, (orig_time, (new_time, _)) in enumerate(sorted_stretch_points):
        new_times.append(new_time)
        original_times.append(orig_time)
    
    # Add the end of the audio if it's not in the stretching map
    if time[-1] > original_times[-1]:
        original_times.append(time[-1])
        new_times.append(new_times[-1] + (time[-1] - original_times[-2]) * stretching_map[sorted_stretch_points[-1][0]][1])
    
    # Create the new time array using piecewise linear interpolation
    new_time = np.interp(time, original_times, new_times)
    
    # Resample the audio
    y_retimed = librosa.resample(y, orig_sr=sr, target_sr=sr * (len(new_time) / len(time)))
    
    # Save the retimed audio
    sf.write(output_wav, y_retimed, sr)

def main():
    parser = argparse.ArgumentParser(description="Retime a WAV file based on MIDI timing.")
    parser.add_argument("input_wav", help="Path to input WAV file")
    parser.add_argument("midi_file", help="Path to MIDI master timing file")
    parser.add_argument("timing_file", help="Path to timing dictionary text file")
    parser.add_argument("output_wav", help="Path to output retimed WAV file")
    args = parser.parse_args()

    wav_timings = parse_timing_dict(args.timing_file)
    midi_timings = get_midi_timings(args.midi_file)
    
    if not wav_timings or not midi_timings:
        print("Error: Empty timing information. Please check your input files.")
        return

    stretching_map = create_stretching_map(wav_timings, midi_timings)
    
    if not stretching_map:
        print("Error: Could not create stretching map. Please check if WAV and MIDI timings are compatible.")
        return

    retime_audio(args.input_wav, stretching_map, args.output_wav)

if __name__ == "__main__":
    main()