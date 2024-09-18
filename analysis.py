import argparse
import json
import mido
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# Constants for colored output
RED = '\033[91m'
WHITE = '\033[97m'
RESET = '\033[0m'

def load_midi(file_path, default_end_time=None):
    mid = mido.MidiFile(file_path)
    notes = []
    current_time = 0
    tempo = 500000  # Default tempo (microseconds per beat)

    for msg in mid:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append((msg.note, current_time, None))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            for i in range(len(notes) - 1, -1, -1):
                if notes[i][0] == msg.note and notes[i][2] is None:
                    notes[i] = (notes[i][0], notes[i][1], current_time)
                    break

    if default_end_time is not None:
        notes = [(note[0], note[1], default_end_time if note[2] is None else note[2]) for note in notes]
    else:
        notes = [note for note in notes if note[2] is not None]

    return notes

def calculate_overlap(notes_a, notes_b):
    overlap = 0
    total_duration_a = sum(note[2] - note[1] for note in notes_a)
    total_duration_b = sum(note[2] - note[1] for note in notes_b)
    total_area = max(total_duration_a, total_duration_b)
    
    for note_a in notes_a:
        for note_b in notes_b:
            if note_a[0] == note_b[0]:  # Same pitch
                start = max(note_a[1], note_b[1])
                end = min(note_a[2], note_b[2])
                if start < end:
                    overlap += end - start
    return overlap / total_area if total_area > 0 else 0

def stretch_notes(notes, stretch_factor):
    return [(note[0], note[1] * stretch_factor, note[2] * stretch_factor) for note in notes]

def optimize_stretch(notes_a, notes_b):
    def objective(stretch_factor):
        stretched_notes_b = stretch_notes(notes_b, stretch_factor)
        overlap = calculate_overlap(notes_a, stretched_notes_b)
        return -overlap
    result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
    return result.x

def recursive_align(notes_analysis, notes_master, timing_dict, start=0, end=None, depth=0, max_depth=10):
    if end is None:
        end = max(note[2] for note in notes_master)
    if (end - start) < 4 or depth > max_depth:
        return
    
    midpoint = (start + end) / 2
    left_analysis = [note for note in notes_analysis if note[1] < midpoint]
    right_analysis = [note for note in notes_analysis if note[2] > midpoint]
    left_master = [note for note in notes_master if note[1] < midpoint]
    right_master = [note for note in notes_master if note[2] > midpoint]
    
    # Move midpoint to the left
    left_midpoint = midpoint * 0.9
    left_stretch_left = optimize_stretch(left_master, [note for note in left_analysis if note[1] < left_midpoint]) if left_analysis and left_master else 1.0
    left_stretch_right = optimize_stretch([note for note in right_master if note[1] < midpoint], [note for note in right_analysis if note[1] < midpoint]) if right_analysis and right_master else 1.0
    left_overlap = calculate_overlap(left_master, stretch_notes(left_analysis, left_stretch_left)) + calculate_overlap([note for note in right_master if note[1] < midpoint], stretch_notes([note for note in right_analysis if note[1] < midpoint], left_stretch_right))
    
    # Move midpoint to the right
    right_midpoint = midpoint * 1.1
    right_stretch_left = optimize_stretch([note for note in left_master if note[2] > midpoint], [note for note in left_analysis if note[2] > midpoint]) if left_analysis and left_master else 1.0
    right_stretch_right = optimize_stretch(right_master, [note for note in right_analysis if note[2] > right_midpoint]) if right_analysis and right_master else 1.0
    right_overlap = calculate_overlap([note for note in left_master if note[2] > midpoint], stretch_notes([note for note in left_analysis if note[2] > midpoint], right_stretch_left)) + calculate_overlap(right_master, stretch_notes([note for note in right_analysis if note[2] > right_midpoint], right_stretch_right))
    
    if left_overlap > right_overlap:
        timing_dict[left_midpoint] = (left_stretch_left + left_stretch_right) / 2
        recursive_align(left_analysis, left_master, timing_dict, start, left_midpoint, depth+1, max_depth)
        recursive_align(right_analysis, right_master, timing_dict, left_midpoint, end, depth+1, max_depth)
    else:
        timing_dict[right_midpoint] = (right_stretch_left + right_stretch_right) / 2
        recursive_align(left_analysis, left_master, timing_dict, start, right_midpoint, depth+1, max_depth)
        recursive_align(right_analysis, right_master, timing_dict, right_midpoint, end, depth+1, max_depth)

def plot_timings(notes_analysis, notes_master, notes_retimed, output_path, master_duration):
    plt.figure(figsize=(20, 10))
    
    analysis_plot = plt.plot([note[1] for note in notes_analysis], [note[0] for note in notes_analysis], 'ro', alpha=0.5)
    master_plot = plt.plot([note[1] for note in notes_master], [note[0] + 0.2 for note in notes_master], 'bo', alpha=0.5)
    retimed_plot = plt.plot([note[1] for note in notes_retimed], [note[0] + 0.4 for note in notes_retimed], 'go', alpha=0.5)

    plt.legend(['Analysis Original', 'Master', 'Analysis Retimed'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('MIDI Timing Alignment')
    plt.xlim(0, master_duration)
    plt.savefig(output_path)
    plt.close()

def main(analysis_midi_path, instrument, master_midi_path, output_json, output_jpg, output_midi):
    # Load MIDI files
    analysis_notes = load_midi(analysis_midi_path)
    master_notes = load_midi(master_midi_path)
    
    if not analysis_notes:
        print(f"{RED}Error: No notes found in analysis MIDI file.{RESET}")
        return
    if not master_notes:
        print(f"{RED}Error: No notes found in master MIDI file.{RESET}")
        return
    
    # Filter master notes based on instrument
    if instrument.lower() == "piano":
        master_notes = [note for note in master_notes if 21 <= note[0] <= 108]  # Piano range
    elif instrument.lower() == "orchestra":
        master_notes = [note for note in master_notes if note[0] < 21 or note[0] > 108]  # Non-piano range
    
    if not master_notes:
        print(f"{RED}Error: No notes found for instrument '{instrument}' in master MIDI file.{RESET}")
        return
    
    master_duration = max(note[2] for note in master_notes)
    analysis_duration = max(note[2] for note in analysis_notes)
    
    # Initial stretch to match durations
    initial_stretch = master_duration / analysis_duration
    analysis_notes = stretch_notes(analysis_notes, initial_stretch)
    
    timing_dict = {0.0: initial_stretch}
    recursive_align(analysis_notes, master_notes, timing_dict, start=0.0, end=master_duration)
    
    # Ensure the last key in timing_dict is master_duration
    if master_duration not in timing_dict:
        timing_dict[master_duration] = 1.0
    
    # Save timing dictionary as JSON
    with open(output_json, 'w') as f:
        json.dump({str(k): v for k, v in sorted(timing_dict.items())}, f, indent=4)
    print(f"Timing adjustments saved to {output_json}")
    
    # Apply timing adjustments to get retimed notes
    retimed_notes = []
    timing_keys = sorted(timing_dict.keys())
    for note in analysis_notes:
        start_idx = np.searchsorted(timing_keys, note[1], side='right') - 1
        end_idx = np.searchsorted(timing_keys, note[2], side='right') - 1
        
        start_time = timing_keys[start_idx]
        end_time = timing_keys[end_idx]
        start_stretch = timing_dict[start_time]
        end_stretch = timing_dict[end_time]
        
        new_start = start_time + (note[1] - start_time) * start_stretch
        new_end = end_time + (note[2] - end_time) * end_stretch
        
        retimed_notes.append((note[0], new_start, new_end))
    
    # Plot timings
    plot_timings(analysis_notes, master_notes, retimed_notes, output_jpg, master_duration)
    print(f"Timing visualization saved to {output_jpg}")

    # Save retimed notes as MIDI
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    current_time = 0
    for note in sorted(retimed_notes, key=lambda x: x[1]):  # Sort notes by start time
        note_start = int(note[1] * 1000)
        note_duration = int((note[2] - note[1]) * 1000)
        
        # Calculate delta time
        delta_time = max(0, note_start - current_time)
        
        track.append(mido.Message('note_on', note=note[0], velocity=64, time=delta_time))
        track.append(mido.Message('note_off', note=note[0], velocity=64, time=note_duration))
        
        current_time = note_start + note_duration

    mid.save(output_midi)
    print(f"Retimed MIDI saved to {output_midi}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and retime MIDI files.')
    parser.add_argument('analysis_midi', help='Path to the analysis MIDI instrument file')
    parser.add_argument('instrument', choices=['piano', 'orchestra'], help='Instrument to analyze')
    parser.add_argument('master_midi', help='Path to the master MIDI file')
    parser.add_argument('output_json', help='Path to output timing JSON file')
    parser.add_argument('output_jpg', help='Path to output visualization JPG file')
    parser.add_argument('output_midi', help='Path to output retimed MIDI file')
    
    args = parser.parse_args()
    main(args.analysis_midi, args.instrument, args.master_midi, args.output_json, args.output_jpg, args.output_midi)