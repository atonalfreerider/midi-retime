import argparse
import json
import mido
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.patches as mpatches

# Add these constants at the beginning of the file
RED = '\033[91m'
WHITE = '\033[97m'
RESET = '\033[0m'

def load_midi(file_path, default_end_time=None):
    mid = mido.MidiFile(file_path)
    tracks_notes = {}
    tempo = 500000  # Move tempo initialization outside the track loop

    for i, track in enumerate(mid.tracks):
        notes = []
        current_time = 0
        # Remove tempo initialization from inside the loop
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo  # Update global tempo
            current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append((msg.note, current_time, None))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                for j in range(len(notes) -1, -1, -1):
                    if notes[j][0] == msg.note and notes[j][2] is None:
                        notes[j] = (notes[j][0], notes[j][1], current_time)
                        break
        # Assign default_end_time to notes without an end time
        if default_end_time is not None:
            for j in range(len(notes)):
                if notes[j][2] is None:
                    notes[j] = (notes[j][0], notes[j][1], default_end_time)
        else:
            # Remove notes without end times
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
    
    # Prevent division by zero or empty lists
    if not left_analysis or not left_master:
        stretch_left = 1.0
    else:
        stretch_left = optimize_stretch(left_master, left_analysis)
    
    if not right_analysis or not right_master:
        stretch_right = 1.0
    else:
        stretch_right = optimize_stretch(right_master, right_analysis)
    
    # Record the stretch factors at the midpoint
    timing_dict[midpoint] = (stretch_left + stretch_right) / 2
    
    # Continue recursion for left and right segments
    recursive_align(left_analysis, left_master, timing_dict, start, midpoint, depth+1, max_depth)
    recursive_align(right_analysis, right_master, timing_dict, midpoint, end, depth+1, max_depth)

def plot_timings(notes_analysis, notes_master, notes_retimed, output_path, master_duration):
    plt.figure(figsize=(20, 10))
    # Plot original analysis notes
    y_shift_analysis = 0.1
    for note in notes_analysis:
        plt.plot([note[1], note[2]], [note[0] + y_shift_analysis, note[0] + y_shift_analysis], color='red', alpha=0.5)

    # Plot master notes
    y_shift_master = 0.2
    for note in notes_master:
        plt.plot([note[1], note[2]], [note[0] + y_shift_master, note[0] + y_shift_master], color='blue', alpha=0.5)

    # Plot retimed analysis notes
    y_shift_retimed = 0.3
    for note in notes_retimed:
        plt.plot([note[1], note[2]], [note[0] + y_shift_retimed, note[0] + y_shift_retimed], color='green', alpha=0.5)

    # Fix legend colors using custom patches to match plotted colors
    analysis_patch = mpatches.Patch(color='red', label='Analysis Original')
    master_patch = mpatches.Patch(color='blue', label='Master')
    retimed_patch = mpatches.Patch(color='green', label='Analysis Retimed')
    plt.legend(handles=[analysis_patch, master_patch, retimed_patch])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('MIDI Timing Alignment')
    plt.xlim(0, master_duration)  # Ensure the x-axis matches the master duration
    plt.savefig(output_path)
    plt.close()

def validate_timing_dict(timing_dict, master_duration):
    # Sort the timing_dict by keys as floats
    sorted_timing = dict(sorted(timing_dict.items(), key=lambda item: float(item[0])))

    # Initialize variables
    previous_time = 0.0
    total_adjusted_duration = 0.0

    # Add an end point at master_duration if not present (use float key)
    if master_duration not in sorted_timing:
        sorted_timing[master_duration] = 1.0  # No stretch at the end

    # Re-sort after adding end point
    sorted_timing = dict(sorted(sorted_timing.items(), key=lambda item: float(item[0])))

    keys = list(sorted_timing.keys())
    stretch_factors = list(sorted_timing.values())

    # Calculate total adjusted duration
    for i in range(len(keys) - 1):
        start = float(keys[i])
        end = float(keys[i + 1])
        stretch = float(stretch_factors[i])
        section_duration = end - start
        adjusted_duration = section_duration * stretch
        total_adjusted_duration += adjusted_duration

    # Validate the total adjusted duration
    if abs(total_adjusted_duration - master_duration) < 0.1:  # Tolerance of 100ms
        print(f"{WHITE}Validation Passed: Adjusted duration matches master duration.{RESET}")
        return True
    else:
        discrepancy = total_adjusted_duration - master_duration
        print(f"{RED}Validation Failed: Adjusted duration ({total_adjusted_duration:.2f}s) does not match master duration ({master_duration:.2f}s). Discrepancy: {discrepancy:.2f}s{RESET}")
        return False

def normalize_timing_dict(timing_dict, master_duration):
    # Calculate current total adjusted duration
    sorted_timing = dict(sorted(timing_dict.items(), key=lambda item: float(item[0])))
    keys = list(sorted_timing.keys())
    stretch_factors = list(sorted_timing.values())
    
    # Add end point if not present (use float key)
    if master_duration not in sorted_timing:
        sorted_timing[master_duration] = 1.0
    
    keys = list(sorted_timing.keys())
    stretch_factors = list(sorted_timing.values())
    
    total_adjusted_duration = 0.0
    durations = []
    stretches = []
    for i in range(len(keys) - 1):
        start = float(keys[i])
        end = float(keys[i + 1])
        stretch = float(stretch_factors[i])
        section_duration = end - start
        adjusted_duration = section_duration * stretch
        durations.append(adjusted_duration)
        stretches.append(stretch)
        total_adjusted_duration += adjusted_duration
    
    # Calculate normalization factor
    normalization_factor = master_duration / total_adjusted_duration if total_adjusted_duration != 0 else 1.0
    
    # Normalize stretch factors
    normalized_timing_dict = {}
    for i in range(len(keys) - 1):
        start = float(keys[i])
        normalized_stretch = stretch_factors[i] * normalization_factor
        normalized_timing_dict[start] = normalized_stretch
    
    return normalized_timing_dict

def main(analysis_midi_path, instrument, master_midi_path, output_json, output_jpg):
    # Load MIDI files
    master_midi = mido.MidiFile(master_midi_path)
    master_duration = master_midi.length
    print(f"Master MIDI Duration: {master_duration} seconds")  # Debug statement

    analysis_tracks = load_midi(analysis_midi_path)
    master_tracks = load_midi(master_midi_path, default_end_time=master_duration)
    
    if not analysis_tracks:
        print(f"{RED}Error: No notes found in analysis MIDI file.{RESET}")
        return
    if not master_tracks:
        print(f"{RED}Error: No notes found in master MIDI file.{RESET}")
        return
    
    analysis_notes = list(analysis_tracks.values())[0]
    master_notes = []
    if instrument.lower() == "piano":
        for track_name, notes in master_tracks.items():
            if "piano" in track_name.lower():
                master_notes.extend(notes)
    elif instrument.lower() == "orchestra":
        for track_name, notes in master_tracks.items():
            if "piano" not in track_name.lower():
                master_notes.extend(notes)
    
    # Log master notes coverage
    max_master_note_time = max(note[2] for note in master_notes) if master_notes else 0
    print(f"Maximum end time in master notes: {max_master_note_time} seconds")
    print(f"Master MIDI Length: {master_duration} seconds")
    
    # Optional: Log the number of notes with missing end times
    missing_end_notes = [note for note in master_notes if note[2] is None]
    if missing_end_notes:
        print(f"{RED}Warning: {len(missing_end_notes)} notes in master MIDI were missing end times and have been set to master_duration.{RESET}")
    
    # Define analysis_duration before initial_stretch
    analysis_duration = max(note[2] for note in analysis_notes)

    # Log analysis_duration
    print(f"Analysis MIDI Duration: {analysis_duration} seconds")

    initial_stretch = master_duration / analysis_duration if analysis_duration > 0 else 1.0
    retimed_notes = stretch_notes(analysis_notes, initial_stretch)
    
    # Initialize timing dictionary
    timing_dict = {0.0: initial_stretch}
    
    # Recursive alignment
    recursive_align(retimed_notes, master_notes, timing_dict, start=0.0, end=master_duration, depth=0, max_depth=10)
    
    # Normalize timing dictionary to ensure total duration matches master_duration
    timing_dict = normalize_timing_dict(timing_dict, master_duration)
    
    # Validate timing dictionary
    if validate_timing_dict(timing_dict, master_duration):
        print(f"{WHITE}Timing dictionary validation succeeded.{RESET}")
    else:
        print(f"{RED}Warning: Timing dictionary validation failed.{RESET}")
    
    # Save timing dictionary
    with open(output_json, 'w') as f:
        # Convert keys back to strings for JSON compatibility
        json.dump({str(k): v for k, v in timing_dict.items()}, f, indent=4)
    print(f"Timing adjustments saved to {output_json}")
    
    # Apply stretch factors to analysis notes
    final_retimed_notes = []
    sorted_timing = dict(sorted(timing_dict.items(), key=lambda item: float(item[0])))
    keys = list(sorted_timing.keys())
    stretch_factors = list(sorted_timing.values())
    
    # Correctly apply stretch factors without altering notes outside segments
    for i in range(len(keys) - 1):
        start = float(keys[i])
        end = float(keys[i + 1])
        stretch = float(stretch_factors[i])
        for note in analysis_notes:
            if start <= note[1] < end and note[2] <= end:
                new_start = start + (note[1] - start) * stretch
                new_end = start + (note[2] - start) * stretch
                final_retimed_notes.append((note[0], new_start, new_end))
            elif note[1] < start and note[2] > end:
                # Note spans the entire segment
                new_start = start + (note[1] - start) * stretch
                new_end = end + (note[2] - end) * stretch
                final_retimed_notes.append((note[0], new_start, new_end))
            elif note[1] < start and start <= note[2] <= end:
                # Note starts before the segment and ends within
                new_end = start + (note[2] - start) * stretch
                final_retimed_notes.append((note[0], note[1], new_end))
            elif start <= note[1] < end and note[2] > end:
                # Note starts within the segment and ends after
                new_start = start + (note[1] - start) * stretch
                new_end = end + (note[2] - end) * stretch
                final_retimed_notes.append((note[0], new_start, note[2]))
    
    # Plot timings
    plot_timings(analysis_notes, master_notes, final_retimed_notes, output_jpg, master_duration)
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