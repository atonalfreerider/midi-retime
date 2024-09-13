import argparse
import mido
import librosa
import numpy as np
import soundfile as sf
from scipy.optimize import minimize_scalar
from pyrubberband import pyrb
import matplotlib.pyplot as plt
import librosa.display


def load_midi(file_path):
    mid = mido.MidiFile(file_path)
    notes = []
    current_time = 0
    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append((msg.note, current_time, None))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            for i in range(len(notes) - 1, -1, -1):
                if notes[i][0] == msg.note and notes[i][2] is None:
                    notes[i] = (notes[i][0], notes[i][1], current_time)
                    break
    return [note for note in notes if note[2] is not None]


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


def align_midi_dtw(notes_a, notes_b):
    # Create time grids for both MIDI files
    end_time = max(max(note[2] for note in notes_a), max(note[2] for note in notes_b))
    time_step = 0.05  # 50 milliseconds
    times = np.arange(0, end_time, time_step)
    
    # Create piano roll representations
    piano_roll_a = create_piano_roll(notes_a, times)
    piano_roll_b = create_piano_roll(notes_b, times)
    
    # Process in chunks to reduce memory usage
    chunk_size = 1000  # Adjust this value based on available memory
    num_chunks = (len(times) + chunk_size - 1) // chunk_size
    
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


def plot_midi_notes(notes_a, notes_b, warped_notes_b, output_path):
    plt.figure(figsize=(20, 10))
    
    # Define vertical offsets for each track
    offset_a, offset_b, offset_warped = 0, 0.2, 0.4

    # Plot notes from MIDI A
    for note in notes_a:
        plt.plot([note[1], note[2]], [note[0] + offset_a, note[0] + offset_a], color='blue', linewidth=2, alpha=0.7)

    # Plot notes from original MIDI B
    for note in notes_b:
        plt.plot([note[1], note[2]], [note[0] + offset_b, note[0] + offset_b], color='magenta', linewidth=2, alpha=0.5)

    # Plot notes from warped MIDI B
    for note in warped_notes_b:
        plt.plot([note[1], note[2]], [note[0] + offset_warped, note[0] + offset_warped], color='green', linewidth=2, alpha=0.5)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('MIDI Note Alignment Visualization')
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='MIDI A'),
        plt.Line2D([0], [0], color='magenta', lw=2, label='MIDI B (Original)'),
        plt.Line2D([0], [0], color='green', lw=2, label='MIDI B (Warped)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Note visualization saved to {output_path}")


def main(midi_a_path, midi_b_path, mp3_c_path, output_midi_path, output_mp3_path, output_image_path, test_mode=False):
    notes_a = load_midi(midi_a_path)
    notes_b = load_midi(midi_b_path)
    
    print("Starting alignment process using Dynamic Time Warping...")
    warped_notes_b, time_map = align_midi_dtw(notes_a, notes_b)
    
    # Create a new MIDI file with warped notes
    new_midi = mido.MidiFile()
    track = mido.MidiTrack()
    new_midi.tracks.append(track)
    
    # Sort notes by start time
    warped_notes_b.sort(key=lambda x: x[1])
    
    # Insert warped notes into the MIDI file
    current_time = 0
    for note in warped_notes_b:
        delta_time = max(0, int((note[1] - current_time) * 1000))
        duration = max(0, int((note[2] - note[1]) * 1000))
        if duration > 0:
            track.append(mido.Message('note_on', note=note[0], velocity=64, time=delta_time))
            track.append(mido.Message('note_off', note=note[0], velocity=64, time=duration))
            current_time = note[2]
    
    new_midi.save(output_midi_path)
    print(f"Warped MIDI saved to {output_midi_path}")
    
    if not test_mode:
        # Stretch the MP3 file using the time map
        print("Stretching MP3 file according to time warp...")
        stretch_audio_with_time_map(mp3_c_path, output_mp3_path, time_map)
        print(f"Stretched MP3 saved to {output_mp3_path}")
    
    # Generate note visualization
    print("Generating note visualization...")
    plot_midi_notes(notes_a, notes_b, warped_notes_b, output_image_path)


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
    import sys
    
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Align and stretch MIDI and MP3 files.')
        parser.add_argument('midi_a', help='Path to the first MIDI file (A)')
        parser.add_argument('midi_b', help='Path to the second MIDI file (B)')
        parser.add_argument('mp3_c', help='Path to the MP3 file (C)')
        parser.add_argument('output_midi', help='Path for the output stretched MIDI file')
        parser.add_argument('output_mp3', help='Path for the output stretched MP3 file')
        parser.add_argument('output_image', help='Path for the output visualization image')

        args = parser.parse_args()

        main(args.midi_a, args.midi_b, args.mp3_c, args.output_midi, args.output_mp3, args.output_image)