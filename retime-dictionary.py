import argparse
import numpy as np
import librosa
import soundfile as sf
import mido
from typing import Dict, Tuple
from scipy import interpolate

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

    # Collect all MIDI events
    events = []
    for track in midi.tracks:
        absolute_tick = 0
        for msg in track:
            absolute_tick += msg.time
            events.append((absolute_tick, msg))

    # Sort events by their absolute tick count
    events.sort(key=lambda x: x[0])

    midi_timings = {1: 0.0}  # Measure 1 always starts at t=0
    current_measure = 1
    current_time = 0.0
    current_ticks = 0
    measure_start_ticks = 0

    for event_ticks, msg in events:
        # Check if we've reached or passed a measure boundary
        while current_ticks - measure_start_ticks >= ticks_per_measure:
            current_measure += 1
            measure_start_time = current_time + ((measure_start_ticks + ticks_per_measure - current_ticks) * tempo) / (1000000 * ticks_per_beat)
            midi_timings[current_measure] = measure_start_time
            measure_start_ticks += ticks_per_measure

        # Process tempo and time signature changes
        if msg.type == 'set_tempo':
            # Calculate time up to this tempo change
            delta_ticks = event_ticks - current_ticks
            current_time += (delta_ticks * tempo) / (1000000 * ticks_per_beat)
            current_ticks = event_ticks
            tempo = msg.tempo
        elif msg.type == 'time_signature':
            # Calculate time up to this time signature change
            delta_ticks = event_ticks - current_ticks
            current_time += (delta_ticks * tempo) / (1000000 * ticks_per_beat)
            current_ticks = event_ticks
            
            # Update time signature and ticks per measure
            time_signature = (msg.numerator, msg.denominator)
            ticks_per_measure = ticks_per_beat * 4 * msg.numerator // msg.denominator
            
            # Reset measure_start_ticks to current_ticks
            measure_start_ticks = current_ticks

            # Increment the measure and set the time in midi_timings
            if current_measure > 1:
                current_measure += 1
                midi_timings[current_measure] = current_time

        # Update current time
        if current_ticks < event_ticks:
            delta_ticks = event_ticks - current_ticks
            current_time += (delta_ticks * tempo) / (1000000 * ticks_per_beat)
            current_ticks = event_ticks

    # Process any remaining ticks after the last event
    while current_ticks - measure_start_ticks >= ticks_per_measure:
        current_measure += 1
        measure_start_time = current_time + ((measure_start_ticks + ticks_per_measure - current_ticks) * tempo) / (1000000 * ticks_per_beat)
        midi_timings[current_measure] = measure_start_time
        measure_start_ticks += ticks_per_measure

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
    
    # Separate the original and new time points
    orig_times, new_times = zip(*sorted_stretch_points)
    new_times = [nt[0] for nt in new_times]  # Extract just the time value, not the tuple
    
    # Create a piecewise linear interpolation function
    time_map = interpolate.interp1d(orig_times, new_times, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Apply the time mapping to all sample times
    new_time = time_map(time)
    
    # Calculate the instantaneous stretch factor for each sample
    stretch_factors = np.diff(new_time) / np.diff(time)
    stretch_factors = np.insert(stretch_factors, 0, stretch_factors[0])  # Pad the first element
    
    # Initialize the output audio array
    y_retimed = np.zeros(int(new_time[-1] * sr))
    
    # Perform the time stretching using a phase vocoder
    for i in range(len(sorted_stretch_points) - 1):
        start_time, (new_start_time, _) = sorted_stretch_points[i]
        end_time, (new_end_time, _) = sorted_stretch_points[i + 1]
        
        # Extract the segment
        segment_mask = (time >= start_time) & (time < end_time)
        segment = y[segment_mask]
        
        # Calculate the average stretch factor for this segment
        avg_stretch = (new_end_time - new_start_time) / (end_time - start_time)
        
        # Time-stretch the segment
        stretched_segment = librosa.effects.time_stretch(segment, rate=1/avg_stretch)
        
        # Calculate the new start and end indices
        new_start_idx = int(new_start_time * sr)
        new_end_idx = int(new_end_time * sr)
        
        # Ensure the stretched segment fits exactly in the allocated space
        stretched_segment = librosa.util.fix_length(stretched_segment, size=new_end_idx - new_start_idx)
        
        # Insert the stretched segment into the output array
        y_retimed[new_start_idx:new_end_idx] = stretched_segment
    
    # Handle the last segment
    last_start_time, (last_new_start_time, _) = sorted_stretch_points[-1]
    last_segment = y[time >= last_start_time]
    last_stretch = (new_time[-1] - last_new_start_time) / (time[-1] - last_start_time)
    stretched_last_segment = librosa.effects.time_stretch(last_segment, rate=1/last_stretch)
    
    last_start_idx = int(last_new_start_time * sr)
    y_retimed[last_start_idx:] = librosa.util.fix_length(stretched_last_segment, size=len(y_retimed) - last_start_idx)
    
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