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
    midi = mido.MidiFile(midi_file)
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # Default tempo (microseconds per beat)
    time_signature = (4, 4)  # Default time signature
    ticks_per_measure = ticks_per_beat * time_signature[0]

    # Merge set_tempo and time_signature messages across tracks
    tempo_time_sig_events = []
    for track in midi.tracks:
        for msg in track:
            if msg.type in ['set_tempo', 'time_signature']:
                tempo_time_sig_events.append((msg.time, msg))
    
    # Sort events by their time
    tempo_time_sig_events.sort(key=lambda x: x[0])

    midi_timings = {}
    current_measure = 1
    current_time = 0.0
    current_ticks = 0

    for event_time, msg in tempo_time_sig_events:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'time_signature':
            time_signature = (msg.numerator, msg.denominator)
            ticks_per_measure = ticks_per_beat * 4 * time_signature[0] // time_signature[1]

        # Process all other MIDI messages up to this point
        while current_ticks < event_time:
            if current_ticks + 1 >= ticks_per_measure:
                # Calculate and store the start time for the current measure
                midi_timings[current_measure] = current_time

                # Calculate time to the next measure
                remaining_ticks = ticks_per_measure - current_ticks
                next_measure_time = current_time + (remaining_ticks * tempo) / (1000000 * ticks_per_beat)

                current_measure += 1
                current_ticks = 0  # Reset current_ticks to 0
                current_time = next_measure_time
            else:
                # Update current time and ticks
                seconds_per_tick = tempo / (1000000 * ticks_per_beat)
                current_time += seconds_per_tick
                current_ticks += 1

    # Process any remaining ticks after the last tempo/time signature event
    while current_ticks < midi.length:
        if current_ticks + 1 >= ticks_per_measure:
            midi_timings[current_measure] = current_time
            remaining_ticks = ticks_per_measure - current_ticks
            next_measure_time = current_time + (remaining_ticks * tempo) / (1000000 * ticks_per_beat)
            current_measure += 1
            current_ticks = 0  # Reset current_ticks to 0
            current_time = next_measure_time
        else:
            seconds_per_tick = tempo / (1000000 * ticks_per_beat)
            current_time += seconds_per_tick
            current_ticks += 1

    # Add the last measure if it's not already included
    if current_measure not in midi_timings:
        midi_timings[current_measure] = current_time

    return midi_timings

def create_stretching_map(wav_timings: Dict[int, float], midi_timings: Dict[int, float]) -> Dict[float, Tuple[float, float]]:
    stretching_map = {}
    wav_measures = sorted(wav_timings.keys())
    midi_measures = sorted(midi_timings.keys())
    
    for wav_measure in wav_measures:
        wav_time = wav_timings[wav_measure]
        
        # Find the corresponding MIDI measure
        midi_measure = next((m for m in midi_measures if m >= wav_measure), None)
        
        if midi_measure is None:
            print(f"Warning: Could not find corresponding MIDI measure for WAV measure {wav_measure}")
            continue
        
        midi_time = midi_timings[midi_measure]
        
        stretching_map[wav_time] = (midi_time, 1)  # We'll calculate stretch factors in retime_audio
    
    return stretching_map


def retime_audio(input_wav: str, stretching_map: Dict[float, Tuple[float, float]], output_wav: str):
    # Load the audio file
    y, sr = librosa.load(input_wav, sr=None)
    
    # Create time array
    time = np.arange(len(y)) / sr
    
    # Sort the stretching map by original time
    sorted_stretch_points = sorted(stretching_map.items())
    
    # Create a new time array for the stretched audio
    new_time = np.zeros_like(time)
    
    # Initialize cumulative stretch factor and time offset
    cumulative_stretch = 1.0
    time_offset = 0.0
    
    # Interpolate the new time array
    for i in range(len(sorted_stretch_points) - 1):
        start_time, (new_start_time, _) = sorted_stretch_points[i]
        end_time, (new_end_time, _) = sorted_stretch_points[i + 1]
        
        # Calculate the stretch factor for this segment
        segment_duration = end_time - start_time
        new_segment_duration = new_end_time - new_start_time
        stretch_factor = new_segment_duration / segment_duration
        
        # Apply the stretch to this segment
        mask = (time >= start_time) & (time < end_time)
        segment_time = time[mask] - start_time
        new_time[mask] = time_offset + (segment_time * stretch_factor * cumulative_stretch)
        
        # Update cumulative stretch and time offset
        time_offset = new_time[mask][-1] if np.any(mask) else time_offset
        cumulative_stretch *= stretch_factor
    
    # Handle the last segment
    last_start_time, (last_new_start_time, _) = sorted_stretch_points[-1]
    mask = time >= last_start_time
    segment_time = time[mask] - last_start_time
    new_time[mask] = time_offset + (segment_time * cumulative_stretch)
    
    # Perform the time stretching using librosa
    y_retimed = librosa.effects.time_stretch(y, rate=len(time)/len(new_time))
    
    # Resample to match the new timing
    target_sr = sr * (len(new_time) / len(time))
    y_retimed = librosa.resample(y_retimed, orig_sr=sr, target_sr=target_sr)
    
    # Ensure the output duration matches the MIDI duration
    target_duration = sorted_stretch_points[-1][1][0]  # Last MIDI time
    current_duration = len(y_retimed) / target_sr
    if not np.isclose(current_duration, target_duration):
        y_retimed = librosa.effects.time_stretch(y_retimed, rate=current_duration / target_duration)
    
    # Save the retimed audio
    sf.write(output_wav, y_retimed, int(target_sr))

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