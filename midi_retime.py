import argparse
import mido
import librosa
import numpy as np
from scipy.optimize import minimize_scalar
from pyrubberband import pyrb
import matplotlib.pyplot as plt
import librosa.display
import shutil

# Add color codes for the status bars
WHITE = '\033[97m'
RED = '\033[91m'
RESET = '\033[0m'

def log_status_bars(step, piano_length, no_piano_length, master_length, scaling_factor):
    def create_bar(length, step, is_master=False):
        if is_master:
            bar = f"{WHITE}{'█' * length}{RESET}"
            return bar
        subdivisions = 2 ** step
        section_length = length // subdivisions
        bar = ""
        for i in range(subdivisions):
            color = WHITE if i % 2 == 0 else RED
            bar += f"{color}{'█' * section_length}{RESET}"
        # Handle any remaining length
        remaining = length - (section_length * subdivisions)
        if remaining > 0:
            bar += WHITE + '█' * remaining + RESET
        return bar

    # Scale lengths based on the fixed scaling_factor
    piano_length_px = int(piano_length * scaling_factor)
    no_piano_length_px = int(no_piano_length * scaling_factor)
    master_length_px = int(master_length * scaling_factor)

    piano_bar = create_bar(piano_length_px, step)
    no_piano_bar = create_bar(no_piano_length_px, step)
    master_bar = create_bar(master_length_px, step, is_master=True)

    # Format durations to mm:ss for display
    def format_duration(seconds):
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    piano_duration_str = format_duration(piano_length)
    no_piano_duration_str = format_duration(no_piano_length)
    master_duration_str = format_duration(master_length)

    print("\nProgress:")
    print(f"Piano Analysis MIDI ({piano_duration_str}):    |{piano_bar}|")
    print(f"No-Piano Analysis MIDI ({no_piano_duration_str}): |{no_piano_bar}|")
    print(f"Master MIDI ({master_duration_str}):           |{master_bar}|")

def load_midi(file_path):
    mid = mido.MidiFile(file_path)
    tracks_notes = {}  # Dictionary to hold notes per track

    for i, track in enumerate(mid.tracks):
        notes = []
        current_time = 0
        tempo = 500000  # Default tempo (microseconds per beat)
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
        # Only keep notes that have both start and end times
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


def stretch_notes(notes, start_time, end_time, stretch_factor):
    stretched_notes = []
    for note in notes:
        if note[1] >= start_time and note[2] <= end_time:
            new_start = start_time + (note[1] - start_time) * stretch_factor
            new_end = start_time + (note[2] - start_time) * stretch_factor
            stretched_notes.append((note[0], new_start, new_end))
        elif note[1] < start_time and note[2] > end_time:
            stretched_notes.append(note)
        elif note[1] < start_time and note[2] > start_time:
            new_end = start_time + (note[2] - start_time) * stretch_factor
            stretched_notes.append((note[0], note[1], new_end))
        elif note[1] < end_time and note[2] > end_time:
            new_start = start_time + (note[1] - start_time) * stretch_factor
            stretched_notes.append((note[0], new_start, note[2]))
    return stretched_notes


def optimize_stretch(notes_a, notes_b, start_time, end_time):
    def objective(stretch_factor):
        stretched_notes_b = stretch_notes(notes_b, start_time, end_time, stretch_factor)
        overlap = calculate_overlap(notes_a, stretched_notes_b)
        return -overlap  # We want to maximize overlap, so we minimize negative overlap

    result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
    return result.x


def align_midi_dtw(notes_a, notes_b, max_duration, scaling_factor):
    # Create time grids for both MIDI files
    end_time = max_duration  # Use the overall maximum duration
    time_step = 0.05  # 50 milliseconds
    times = np.arange(0, end_time, time_step)
    
    # Create piano roll representations
    piano_roll_a = create_piano_roll(notes_a, times)
    piano_roll_b = create_piano_roll(notes_b, times)
    
    # Process in chunks to reduce memory usage
    chunk_size = 1000  # Adjust this value based on available memory
    num_chunks = (len(times) + chunk_size - 1) // chunk_size
    
    max_steps = int(np.log2(num_chunks)) if num_chunks > 0 else 1
    warped_times = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(times))
        
        chunk_a = piano_roll_a[start_idx:end_idx]
        chunk_b = piano_roll_b[start_idx:end_idx]
        
        # Compute DTW for the chunk
        D, wp = librosa.sequence.dtw(X=chunk_a.T, Y=chunk_b.T, metric='euclidean')
        
        # Extract the warp path for this chunk
        path_x = wp[:, 0] + start_idx
        path_y = wp[:, 1] + start_idx
        
        warped_times.extend(zip(times[path_y], times[path_x]))
    
    # Create a mapping from time in B to time in A
    time_map = dict(warped_times)
    
    # Apply the time warping to notes_b
    warped_notes_b = warp_notes(notes_b, time_map)
    
    return warped_notes_b, time_map

def create_piano_roll(notes, times):
    # MIDI notes range from 0 to 127
    piano_roll = np.zeros((len(times), 128), dtype=np.uint8)
    for note in notes:
        start_idx = np.searchsorted(times, note[1])
        end_idx = np.searchsorted(times, note[2])
        piano_roll[start_idx:end_idx, note[0]] = 1
    return piano_roll

def warp_notes(notes, time_map):
    warped_notes = []
    time_steps = sorted(time_map.keys())
    for note in notes:
        # Map start and end times using the time_map
        warped_start = interpolate_time(note[1], time_steps, time_map)
        warped_end = interpolate_time(note[2], time_steps, time_map)
        warped_notes.append((note[0], warped_start, warped_end))
    return warped_notes

def interpolate_time(time, time_steps, time_map):
    # Find the closest time steps before and after the time
    idx = np.searchsorted(time_steps, time)
    if idx == 0:
        return time_map[time_steps[0]]
    elif idx == len(time_steps):
        return time_map[time_steps[-1]]
    else:
        t_before = time_steps[idx - 1]
        t_after = time_steps[idx]
        mapped_t_before = time_map[t_before]
        mapped_t_after = time_map[t_after]
        # Linear interpolation
        proportion = (time - t_before) / (t_after - t_before)
        return mapped_t_before + proportion * (mapped_t_after - mapped_t_before)


def shift_notes(notes, offset):
    shifted_notes = []
    for note in notes:
        shifted_notes.append((note[0], note[1] + offset, note[2] + offset))
    return shifted_notes


def apply_stretching(notes, stretches):
    stretched_notes = []
    for note in notes:
        new_start = note[1]
        new_end = note[2]
        for start, end, factor in stretches:
            if start <= new_start < end:
                new_start = start + (new_start - start) * factor
            if start < new_end <= end:
                new_end = start + (new_end - start) * factor
        stretched_notes.append((note[0], new_start, new_end))
    return stretched_notes


def stretch_audio(y, sr, stretches):
    stretched_audio = []
    for start, end, factor in stretches:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        stretched_segment = pyrb.time_stretch(segment, sr, factor)
        stretched_audio.append(stretched_segment)
    return np.concatenate(stretched_audio)


def plot_midi_notes(master_piano_notes, analysis_piano_notes, warped_analysis_piano_notes, output_path):
    plt.figure(figsize=(20, 10))
    
    # Define vertical offsets for each track
    offset_master_piano, offset_analysis_piano, offset_warped_analysis_piano = 0, 0.2, 0.4

    # Plot notes from Master Piano
    for note in master_piano_notes:
        plt.plot([note[1], note[2]], [note[0] + offset_master_piano, note[0] + offset_master_piano], color='blue', linewidth=2, alpha=0.7)

    # Plot notes from Analysis Piano
    for note in analysis_piano_notes:
        plt.plot([note[1], note[2]], [note[0] + offset_analysis_piano, note[0] + offset_analysis_piano], color='magenta', linewidth=2, alpha=0.5)

    # Plot warped Analysis Piano notes
    for note in warped_analysis_piano_notes:
        plt.plot([note[1], note[2]], [note[0] + offset_warped_analysis_piano, note[0] + offset_warped_analysis_piano], color='green', linewidth=2, alpha=0.5)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('Piano MIDI Note Alignment Visualization')
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Master Piano'),
        plt.Line2D([0], [0], color='magenta', lw=2, label='Analysis Piano (Original)'),
        plt.Line2D([0], [0], color='green', lw=2, label='Analysis Piano (Warped)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Piano note visualization saved to {output_path}")

def plot_midi_notes_no_piano(master_orchestra_notes, analysis_orchestra_notes, warped_analysis_orchestra_notes, output_path):
    plt.figure(figsize=(20, 10))
    
    # Define vertical offsets for each track
    offset_master_orchestra, offset_analysis_orchestra, offset_warped_analysis_orchestra = 0, 0.2, 0.4

    # Plot notes from Master Orchestra
    for note in master_orchestra_notes:
        plt.plot([note[1], note[2]], [note[0] + offset_master_orchestra, note[0] + offset_master_orchestra], color='blue', linewidth=2, alpha=0.7)

    # Plot notes from Analysis Orchestra
    for note in analysis_orchestra_notes:
        plt.plot([note[1], note[2]], [note[0] + offset_analysis_orchestra, note[0] + offset_analysis_orchestra], color='magenta', linewidth=2, alpha=0.5)

    # Plot warped Analysis Orchestra notes
    for note in warped_analysis_orchestra_notes:
        plt.plot([note[1], note[2]], [note[0] + offset_warped_analysis_orchestra, note[0] + offset_warped_analysis_orchestra], color='green', linewidth=2, alpha=0.5)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('Orchestra MIDI Note Alignment Visualization')
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Master Orchestra'),
        plt.Line2D([0], [0], color='magenta', lw=2, label='Analysis Orchestra (Original)'),
        plt.Line2D([0], [0], color='green', lw=2, label='Analysis Orchestra (Warped)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Orchestra note visualization saved to {output_path}")

def main(analysis_piano_midi_path, analysis_no_piano_midi_path, master_midi_path, mp3_c_path, output_midi_path, output_mp3_path, output_image_piano_path, output_image_orchestra_path, test_mode=False):
    # Load analysis MIDI files
    analysis_piano_tracks = load_midi(analysis_piano_midi_path)
    analysis_no_piano_tracks = load_midi(analysis_no_piano_midi_path)

    # Load Master MIDI file
    master_tracks = load_midi(master_midi_path)

    # Extract notes for comparison
    # Analysis piano notes (from the only track in the piano MIDI)
    analysis_piano_notes = list(analysis_piano_tracks.values())[0]

    # Analysis orchestra notes (from the only track in the no_piano MIDI)
    analysis_orchestra_notes = list(analysis_no_piano_tracks.values())[0]

    # Master piano notes
    master_piano_notes = []
    piano_track_names = []
    for track_name, track_events in master_tracks.items():
        # Check if the track name contains 'piano' (case-insensitive)
        if 'piano' in track_name.lower():
            master_piano_notes.extend(track_events)
            piano_track_names.append(track_name)

    if not master_piano_notes:
        print("No piano tracks found in Master MIDI.")

    # Master orchestra notes (exclude piano tracks)
    orchestra_track_names = [name for name in master_tracks.keys() if name not in piano_track_names]
    master_orchestra_notes = []
    for track_name in orchestra_track_names:
        master_orchestra_notes.extend(master_tracks[track_name])

    # Calculate durations in seconds
    piano_duration = max(note[2] for note in analysis_piano_notes) if analysis_piano_notes else 0
    no_piano_duration = max(note[2] for note in analysis_orchestra_notes) if analysis_orchestra_notes else 0
    master_duration = max(note[2] for note in master_piano_notes + master_orchestra_notes) if (master_piano_notes + master_orchestra_notes) else 0

    # Determine the maximum duration among all tracks
    max_duration = max(piano_duration, no_piano_duration, master_duration)    

    # Get terminal width
    terminal_width = shutil.get_terminal_size(fallback=(80, 20)).columns

    # Define labels for each MIDI type
    labels = [
        "Piano Analysis MIDI:",
        "No-Piano Analysis MIDI:",
        "Master MIDI:"
    ]
    
    # Calculate the maximum label length
    max_label_length = max(len(label) for label in labels) + 10  # Extra space for duration and pipe
    
    # Define padding and separators
    padding = 2  # For spacing and pipe characters
    max_bar_length = terminal_width - max_label_length - padding
    max_bar_length = max_bar_length if max_bar_length > 0 else 50  # Fallback to 50 if negative

    # Calculate fixed scaling factor based on the maximum duration
    scaling_factor = max_bar_length / max_duration if max_duration > 0 else 1

    # Initialize combined time map
    combined_time_map = {}

    # Initialize warped notes
    warped_notes_piano = analysis_piano_notes
    warped_notes_orchestra = analysis_orchestra_notes

    # Determine number of alignment passes
    num_passes = 10  # Define how many alignment passes to perform

    for pass_num in range(1, num_passes + 1):
        if pass_num % 2 == 1:
            # Odd pass: align piano
            print(f"Alignment Pass {pass_num}: Aligning Piano")
            warped_notes_piano, time_map_piano = align_midi_dtw(master_piano_notes, warped_notes_piano, max_duration, scaling_factor)
            # Update combined_time_map
            combined_time_map = combine_time_maps(combined_time_map, time_map_piano)
        else:
            # Even pass: align no-piano
            print(f"Alignment Pass {pass_num}: Aligning No-Piano")
            warped_notes_orchestra, time_map_orchestra = align_midi_dtw(master_orchestra_notes, warped_notes_orchestra, max_duration, scaling_factor)
            # Update combined_time_map
            combined_time_map = combine_time_maps(combined_time_map, time_map_orchestra)
        
        # Apply the combined time map to warp all analysis notes
        warped_notes_piano = warp_notes(warped_notes_piano, combined_time_map)
        warped_notes_orchestra = warp_notes(warped_notes_orchestra, combined_time_map)

        # Log status bars with fixed scaling_factor and actual durations
        current_piano_duration = max(note[2] for note in warped_notes_piano) if warped_notes_piano else 0
        current_no_piano_duration = max(note[2] for note in warped_notes_orchestra) if warped_notes_orchestra else 0
        current_master_duration = master_duration  # Assuming master is fixed

        log_status_bars(
            step=pass_num,
            piano_length=current_piano_duration,
            no_piano_length=current_no_piano_duration,
            master_length=current_master_duration,
            scaling_factor=scaling_factor
        )

        # Generate separate visualizations after each alignment pass
        plot_midi_notes(
            master_piano_notes,
            warped_notes_piano,
            warped_notes_piano,  # Pass warped notes
            output_image_piano_path
        )
        plot_midi_notes_no_piano(
            master_orchestra_notes,
            warped_notes_orchestra,
            warped_notes_orchestra,  # Pass warped notes
            output_image_orchestra_path
        )
    
    # Combine all warped notes
    analysis_notes_combined = warped_notes_piano + warped_notes_orchestra

    # Create a new MIDI file with warped notes
    new_midi = mido.MidiFile()
    # Reconstruct tracks based on analysis MIDI files
    # (Add code here to create tracks and add warped notes)

    # Save the new MIDI file
    new_midi.save(output_midi_path)
    print(f"Warped MIDI saved to {output_midi_path}")

    if not test_mode:
        # Stretch the MP3 file using the combined time map
        print("Stretching MP3 file according to combined time warp...")
        stretch_audio_with_time_map(mp3_c_path, output_mp3_path, combined_time_map)
        print(f"Stretched MP3 saved to {output_mp3_path}")

    # Generate final note visualizations
    print("Generating final note visualizations...")
    plot_midi_notes(
        master_piano_notes,
        warped_notes_piano,
        warped_notes_piano,  # Pass final warped notes
        output_image_piano_path
    )
    plot_midi_notes_no_piano(
        master_orchestra_notes,
        warped_notes_orchestra,
        warped_notes_orchestra,  # Pass final warped notes
        output_image_orchestra_path
    )

    print("Processing complete.")


def combine_time_maps(time_map1, time_map2):
    combined_time_map = {}
    time_keys = sorted(set(time_map1.keys()).union(time_map2.keys()))
    for t in time_keys:
        mapped_times = []
        if t in time_map1:
            mapped_times.append(time_map1[t])
        if t in time_map2:
            mapped_times.append(time_map2[t])
        combined_time_map[t] = sum(mapped_times) / len(mapped_times)
    return combined_time_map


def stretch_audio_with_time_map(input_audio_path, output_audio_path, time_map):
    import soundfile as sf
    y, sr = librosa.load(input_audio_path)
    
    # Generate the time warp function
    original_times = np.array(sorted(time_map.keys()))
    warped_times = np.array([time_map[t] for t in original_times])
    
    # Create an interpolation function for the time warp
    from scipy.interpolate import interp1d
    interp_function = interp1d(warped_times, original_times, kind='linear', fill_value="extrapolate")
    
    # Generate the new time indices for the audio signal
    duration = warped_times[-1]
    new_times = np.arange(0, duration, 1 / sr)
    original_times_mapped = interp_function(new_times)
    
    # Resample the audio signal
    warped_audio = np.interp(original_times_mapped, np.arange(len(y)) / sr, y)
    
    sf.write(output_audio_path, warped_audio, sr)
    print(f"Audio stretched and saved to {output_audio_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align and stretch MIDI and MP3 files.')
    parser.add_argument('analysis_piano_midi', help='Path to the analysis piano MIDI file')
    parser.add_argument('analysis_no_piano_midi', help='Path to the analysis no_piano MIDI file')
    parser.add_argument('master_midi', help='Path to the Master MIDI file')
    parser.add_argument('mp3_c', help='Path to the MP3 file (C)')
    parser.add_argument('output_midi', help='Path for the output stretched MIDI file')
    parser.add_argument('output_mp3', help='Path for the output stretched MP3 file')
    parser.add_argument('output_image_piano', help='Path for the output piano visualization image')
    parser.add_argument('output_image_orchestra', help='Path for the output orchestra visualization image')

    args = parser.parse_args()

    main(args.analysis_piano_midi, args.analysis_no_piano_midi, args.master_midi, args.mp3_c, args.output_midi, args.output_mp3, args.output_image_piano, args.output_image_orchestra)