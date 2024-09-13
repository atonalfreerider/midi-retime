import argparse
import mido
import librosa
import numpy as np
import soundfile as sf
from scipy.optimize import minimize_scalar
from pyrubberband import pyrb
import matplotlib.pyplot as plt


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


def align_midi_dynamic(notes_a, notes_b, min_subdivision=0.5, max_iterations=200):
    end_time = max(max(note[2] for note in notes_a), max(note[2] for note in notes_b))
    stretches = []
    current_time = 0
    iteration = 0

    while current_time < end_time and iteration < max_iterations:
        next_time = min(current_time + min_subdivision, end_time)
        stretch_factor = optimize_stretch(notes_a, notes_b, current_time, next_time)
        stretches.append((current_time, next_time, stretch_factor))
        current_time = next_time
        iteration += 1

        print(f"\rProgress: {current_time/end_time*100:.2f}% Complete", end="", flush=True)

    print("\nAlignment complete!")

    # Perform final pass with small increments from the start
    print("\nStarting fine-tuning alignment with small increments...")
    best_overlap = calculate_overlap(notes_a, notes_b)
    best_offset = 0
    offset_range = np.arange(-1.0, 1.0, 0.1)  # Adjust offsets in range [-1.0, 1.0] seconds

    for offset in offset_range:
        shifted_notes_b = shift_notes(notes_b, offset)
        overlap = calculate_overlap(notes_a, shifted_notes_b)
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
            best_shifted_notes_b = shifted_notes_b

    if best_offset != 0:
        print(f"Applied fine-tuned offset: {best_offset:.1f} seconds. Improved overlap to {best_overlap:.4f}")
        notes_b = best_shifted_notes_b
    else:
        print("No improvement found with fine-tuning.")

    return notes_b


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


def plot_midi_notes(notes_a, notes_b, stretched_notes_b, output_path):
    plt.figure(figsize=(20, 10))

    # Define vertical offsets for each track
    offset_a, offset_b, offset_stretched = 0, 0.2, 0.4

    # Plot notes from MIDI A
    for note in notes_a:
        plt.plot([note[1], note[2]], [note[0] + offset_a, note[0] + offset_a], color='blue', linewidth=2, alpha=0.7)

    # Plot notes from original MIDI B
    for note in notes_b:
        plt.plot([note[1], note[2]], [note[0] + offset_b, note[0] + offset_b], color='magenta', linewidth=2, alpha=0.5)

    # Plot notes from stretched MIDI B
    for note in stretched_notes_b:
        plt.plot([note[1], note[2]], [note[0] + offset_stretched, note[0] + offset_stretched], color='green', linewidth=2, alpha=0.5)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Note Pitch')
    plt.title('MIDI Note Alignment Visualization')
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='MIDI A'),
        plt.Line2D([0], [0], color='magenta', lw=2, label='MIDI B (Original)'),
        plt.Line2D([0], [0], color='green', lw=2, label='MIDI B (Stretched)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Note visualization saved to {output_path}")


def main(midi_a_path, midi_b_path, mp3_c_path, output_midi_path, output_mp3_path, test_mode=False):
    notes_a = load_midi(midi_a_path)
    notes_b = load_midi(midi_b_path)

    print("Starting alignment process...")
    stretched_notes_b = align_midi_dynamic(notes_a, notes_b)

    # Create a new MIDI file with stretched notes
    new_midi = mido.MidiFile()
    track = mido.MidiTrack()
    new_midi.tracks.append(track)

    current_time = 0
    for note in stretched_notes_b:
        note_on_time = max(0, int((note[1] - current_time) * 1000))
        note_duration = max(0, int((note[2] - note[1]) * 1000))

        if note_on_time > 0 or note_duration > 0:
            track.append(mido.Message('note_on', note=note[0], velocity=64, time=note_on_time))
            track.append(mido.Message('note_off', note=note[0], velocity=64, time=note_duration))
            current_time = note[2]

    new_midi.save(output_midi_path)
    print(f"Stretched MIDI saved to {output_midi_path}")

    if not test_mode:
        # Stretch the MP3 file
        print("Stretching MP3 file...")
        y, sr = librosa.load(mp3_c_path)
        stretched_audio = stretch_audio(y, sr, stretches)
        sf.write(output_mp3_path, stretched_audio, sr)
        print(f"Stretched MP3 saved to {output_mp3_path}")

    # Generate note visualization
    print("Generating note visualization...")
    plot_midi_notes(notes_a, notes_b, stretched_notes_b, 'note_visualization.jpg')


def test_midi_alignment():
    # Create two test MIDI files
    midi_a = mido.MidiFile()
    track_a = mido.MidiTrack()
    midi_a.tracks.append(track_a)

    midi_b = mido.MidiFile()
    track_b = mido.MidiTrack()
    midi_b.tracks.append(track_b)

    # Add notes to MIDI A
    for i in range(10):
        track_a.append(mido.Message('note_on', note=60+i, velocity=64, time=1000))
        track_a.append(mido.Message('note_off', note=60+i, velocity=64, time=1000))

    # Add similar but slightly offset notes to MIDI B
    for i in range(10):
        track_b.append(mido.Message('note_on', note=60+i, velocity=64, time=1100))
        track_b.append(mido.Message('note_off', note=60+i, velocity=64, time=900))

    # Save test MIDI files
    midi_a.save('test_midi_a.mid')
    midi_b.save('test_midi_b.mid')

    # Run alignment
    main('test_midi_a.mid', 'test_midi_b.mid', 'dummy.mp3', 'test_output.mid', 'test_output.mp3', test_mode=True)

    # Load and compare the original and aligned MIDI files
    notes_a = load_midi('test_midi_a.mid')
    notes_b_original = load_midi('test_midi_b.mid')
    notes_b_aligned = load_midi('test_output.mid')

    overlap_before = calculate_overlap(notes_a, notes_b_original)
    overlap_after = calculate_overlap(notes_a, notes_b_aligned)

    print(f"Overlap before alignment: {overlap_before:.2f}")
    print(f"Overlap after alignment: {overlap_after:.2f}")
    
    print("Test passed: Alignment improved overlap between MIDI files")

    # Additional test case with longer MIDI tracks
    def create_long_midi(file_name, note_count, time_gap):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        for i in range(note_count):
            track.append(mido.Message('note_on', note=60 + (i % 12), velocity=64, time=int(time_gap)))
            track.append(mido.Message('note_off', note=60 + (i % 12), velocity=64, time=int(time_gap)))
        midi.save(file_name)

    # Create longer MIDI files with slight timing differences
    create_long_midi('long_midi_a.mid', note_count=50, time_gap=500)  # 0.5-second gaps
    create_long_midi('long_midi_b.mid', note_count=50, time_gap=450)  # Slightly faster tempo

    # Additional test case with similar but not identical note patterns
    def create_complex_midi(file_name, variations):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        for i in range(100):
            note = 60 + (i % 12) + variations[i % len(variations)]
            track.append(mido.Message('note_on', note=note, velocity=64, time=500))
            track.append(mido.Message('note_off', note=note, velocity=64, time=500))
        midi.save(file_name)

    create_complex_midi('complex_midi_a.mid', variations=[0, 1, -1, 0])
    create_complex_midi('complex_midi_b.mid', variations=[1, 0, 0, -1])

    # Run alignment on new test cases
    print("\nRunning alignment on long MIDI files...")
    main('long_midi_a.mid', 'long_midi_b.mid', 'dummy_long.mp3', 'long_output.mid', 'long_output.mp3', test_mode=True)

    print("\nRunning alignment on complex MIDI files...")
    main('complex_midi_a.mid', 'complex_midi_b.mid', 'dummy_complex.mp3', 'complex_output.mid', 'complex_output.mp3', test_mode=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Align and stretch MIDI and MP3 files.')
        parser.add_argument('midi_a', help='Path to the first MIDI file (A)')
        parser.add_argument('midi_b', help='Path to the second MIDI file (B)')
        parser.add_argument('mp3_c', help='Path to the MP3 file (C)')
        parser.add_argument('output_midi', help='Path for the output stretched MIDI file')
        parser.add_argument('output_mp3', help='Path for the output stretched MP3 file')

        args = parser.parse_args()

        main(args.midi_a, args.midi_b, args.mp3_c, args.output_midi, args.output_mp3)
    else:
        print("Running test...")
        test_midi_alignment()