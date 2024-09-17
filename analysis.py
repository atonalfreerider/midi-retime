import argparse
import json
import mido
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def load_midi(file_path):
    mid = mido.MidiFile(file_path)
    tracks_notes = {}
    for i, track in enumerate(mid.tracks):
        notes = []
        current_time = 0
        tempo = 500000
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append((msg.note, current_time, None))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for j in range(len(notes) -1, -1, -1):
                    if notes[j][0] == msg.note and notes[j][2] is None:
                        notes[j] = (notes[j][0], notes[j][1], current_time)
                        break
        notes = [note for note in notes if note[2] is not None]
        if notes:
            track_name = track.name if track.name else f'Track_{i}'
            tracks_notes[track_name] = notes
    return tracks_notes

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
    stretched_notes = []
    for note in notes:
        new_start = note[1] * stretch_factor
        new_end = note[2] * stretch_factor
        stretched_notes.append((note[0], new_start, new_end))
    return stretched_notes

def optimize_stretch(notes_a, notes_b):
    def objective(stretch_factor):
        stretched_notes_b = stretch_notes(notes_b, stretch_factor)
        overlap = calculate_overlap(notes_a, stretched_notes_b)
        return -overlap
    result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
    return result.x

def recursive_align(notes_analysis, notes_master, timing_dict, start=0, end=None, depth=0):
    if end is None:
        end = max(note[2] for note in notes_master)
    if (end - start) < 4 or depth > 10:
        return
    midpoint = (start + end) / 2
    left_analysis = [note for note in notes_analysis if note[1] < midpoint]
    right_analysis = [note for note in notes_analysis if note[2] > midpoint]
    left_master = [note for note in notes_master if note[1] < midpoint]
    right_master = [note for note in notes_master if note[2] > midpoint]
    stretch_left = optimize_stretch(left_master, left_analysis)
    stretch_right = optimize_stretch(right_master, right_analysis)
    timing_dict[midpoint] = (stretch_left + stretch_right) / 2
    recursive_align(left_analysis, left_master, timing_dict, start, midpoint, depth+1)
    recursive_align(right_analysis, right_master, timing_dict, midpoint, end, depth+1)

def plot_timings(notes_analysis, notes_master, notes_retimed, output_path):
    plt.figure(figsize=(20, 10))
    # Plot original analysis notes
    for note in notes_analysis:
        plt.plot([note[1], note[2]], [note[0], note[0]], color='red', alpha=0.5)
    # Plot master notes
    for note in notes_master:
        plt.plot([note[1], note[2]], [note[0], note[0]], color='blue', alpha=0.5)
    # Plot retimed analysis notes
    for note in notes_retimed:
        plt.plot([note[1], note[2]], [note[0], note[0]], color='green', alpha=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('MIDI Timing Alignment')
    plt.legend(['Analysis Original', 'Master', 'Analysis Retimed'])
    plt.savefig(output_path)
    plt.close()

def main(analysis_midi_path, instrument, master_midi_path, output_json, output_jpg):
    # Load MIDI files
    analysis_tracks = load_midi(analysis_midi_path)
    master_tracks = load_midi(master_midi_path)
    analysis_notes = list(analysis_tracks.values())[0]
    master_notes = list(master_tracks.values())[0]
    
    # Initial stretching
    master_duration = max(note[2] for note in master_notes)
    analysis_duration = max(note[2] for note in analysis_notes)
    initial_stretch = master_duration / analysis_duration
    retimed_notes = stretch_notes(analysis_notes, initial_stretch)
    
    # Initialize timing dictionary
    timing_dict = {0: initial_stretch}
    
    # Recursive alignment
    recursive_align(retimed_notes, master_notes, timing_dict)
    
    # Save timing dictionary
    with open(output_json, 'w') as f:
        json.dump(timing_dict, f, indent=4)
    
    # Plot timings
    plot_timings(analysis_notes, master_notes, retimed_notes, output_jpg)
    print(f"Timing adjustments saved to {output_json}")
    print(f"Timing visualization saved to {output_jpg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and retime MIDI files.')
    parser.add_argument('analysis_midi', help='Path to the analysis MIDI instrument file')
    parser.add_argument('instrument', choices=['piano', 'orchestra'], help='Instrument to analyze')
    parser.add_argument('master_midi', help='Path to the master MIDI file')
    parser.add_argument('output_json', help='Path to output timing JSON file')
    parser.add_argument('output_jpg', help='Path to output visualization JPG file')
    
    args = parser.parse_args()
    main(args.analysis_midi, args.instrument, args.master_midi, args.output_json, args.output_jpg)