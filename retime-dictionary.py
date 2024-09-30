import argparse
import numpy as np
import librosa
import soundfile as sf
import mido
from typing import Dict, Tuple
import matplotlib.pyplot as plt

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

    last_measure_ticks = 0
    for total_ticks, msg in all_messages:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'time_signature':
            old_time_signature = time_signature
            time_signature = (msg.numerator, msg.denominator)
            old_ticks_per_measure = ticks_per_measure
            ticks_per_measure = ticks_per_beat * 4 * time_signature[0] // time_signature[1]
            # Adjust the current measure based on the time signature change
            measure_progress = (total_ticks - last_measure_ticks) / old_ticks_per_measure
            current_measure += int(measure_progress)
            last_measure_ticks = total_ticks

        # Calculate the current measure
        measure_progress = (total_ticks - last_measure_ticks) / ticks_per_measure
        while measure_progress >= 1:
            current_measure += 1
            last_measure_ticks += ticks_per_measure
            measure_progress -= 1

        # Convert ticks to seconds
        seconds = mido.tick2second(total_ticks, ticks_per_beat, tempo)

        # Update midi_timings if this is a new measure or later timing for the same measure
        if current_measure not in midi_timings or seconds > midi_timings[current_measure]:
            midi_timings[current_measure] = seconds

    # Ensure we capture the last measure and the total duration
    total_duration_seconds = mido.tick2second(total_duration_ticks, ticks_per_beat, tempo)
    last_measure = max(midi_timings.keys())
    if total_duration_seconds > midi_timings[last_measure]:
        last_measure += 1
        midi_timings[last_measure] = total_duration_seconds

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

def create_timing_graph(midi_timings: Dict[int, float], wav_timings: Dict[int, float], output_jpg: str):
    plt.figure(figsize=(12, 8))
    
    # Set up the plot
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_ylim(max(max(midi_timings.values()), max(wav_timings.values())) * 1.1, 0)
    
    # Plot MIDI timings
    midi_measures = sorted(midi_timings.keys())
    midi_times = [midi_timings[m] for m in midi_measures]
    plt.plot([-0.2] * len(midi_times), midi_times, 'r-', label='MIDI')
    
    # Plot WAV timings
    wav_measures = sorted(wav_timings.keys())
    wav_times = [wav_timings[m] for m in wav_measures]
    plt.plot([1.2] * len(wav_times), wav_times, 'g-', label='WAV')
    
    # Add hash marks and labels only for WAV measures (from txt file)
    for measure in wav_measures:
        midi_time = midi_timings.get(measure, None)
        wav_time = wav_timings[measure]
        
        # MIDI hash mark and label (only if measure exists in MIDI)
        if midi_time is not None:
            plt.plot([-0.25, -0.15], [midi_time, midi_time], 'r-')
            plt.text(-0.3, midi_time, f'M{measure}', ha='right', va='center')
        
        # WAV hash mark and label
        plt.plot([1.15, 1.25], [wav_time, wav_time], 'g-')
        plt.text(1.3, wav_time, f'M{measure}', ha='left', va='center')
        
        # Draw skew line and add ratio label
        if midi_time is not None:
            plt.plot([-0.2, 1.2], [midi_time, wav_time], 'm-', alpha=0.5)
            
            # Calculate and display the ratio
            ratio = midi_time / wav_time if wav_time != 0 else float('inf')
            mid_x = 0.5
            mid_y = (midi_time + wav_time) / 2
            plt.text(mid_x, mid_y, f'{ratio:.2f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title('MIDI vs WAV Timing Comparison')
    plt.legend(loc='lower right')
    plt.axis('off')
    
    # Save the graph
    plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
    plt.close()

def retime_audio(input_wav: str, stretching_map: Dict[float, Tuple[float, float]], output_wav: str):
    # Load the audio file
    y, sr = librosa.load(input_wav, sr=None)
    
    # Create time array
    time = np.arange(len(y)) / sr
    
    # Sort the stretching map by original time
    sorted_stretch_points = sorted(stretching_map.items())
    
    # Create a new time array for the stretched audio
    new_time = np.zeros_like(time)
    
    # Interpolate the new time array
    for i in range(len(sorted_stretch_points) - 1):
        start_time, (new_start_time, _) = sorted_stretch_points[i]
        end_time, (new_end_time, _) = sorted_stretch_points[i + 1]
        
        mask = (time >= start_time) & (time < end_time)
        new_time[mask] = np.interp(time[mask], [start_time, end_time], [new_start_time, new_end_time])
    
    # Handle the last segment
    last_start_time, (last_new_start_time, _) = sorted_stretch_points[-1]
    mask = time >= last_start_time
    new_time[mask] = np.interp(time[mask], [last_start_time, time[-1]], [last_new_start_time, new_time[-1]])
    
    # Calculate the new sample rate
    target_sr = sr * (len(new_time) / len(time))
    
    # Perform the time stretching using resample
    y_retimed = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
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

    # Generate the timing graph
    output_jpg = args.output_wav.rsplit('.', 1)[0] + '_timing_graph.jpg'
    create_timing_graph(midi_timings, wav_timings, output_jpg)
    print(f"Timing graph saved as {output_jpg}")

    retime_audio(args.input_wav, stretching_map, args.output_wav)

if __name__ == "__main__":
    main()