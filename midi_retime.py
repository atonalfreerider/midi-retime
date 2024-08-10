import argparse
import mido
import librosa
import numpy as np
import soundfile as sf
from scipy.optimize import minimize_scalar
from pyrubberband import pyrb
import time


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
    total_area = 0
    for note_a in notes_a:
        note_a_area = note_a[2] - note_a[1]
        total_area += note_a_area
        for note_b in notes_b:
            if note_a[0] == note_b[0]:  # Same pitch
                start = max(note_a[1], note_b[1])
                end = min(note_a[2], note_b[2])
                if start < end:
                    overlap += end - start
    return overlap, total_area - overlap


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
        _, non_overlap = calculate_overlap(notes_a, stretched_notes_b)
        return non_overlap

    result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
    return result.x


def align_midi_recursive(notes_a, notes_b, start_time, end_time, min_subdivision=4, depth=0):
    global iteration_count, total_iterations, start_time_global

    if end_time - start_time <= min_subdivision:
        iteration_count += 1
        stretch_factor = optimize_stretch(notes_a, notes_b, start_time, end_time)
        stretched_notes_b = stretch_notes(notes_b, start_time, end_time, stretch_factor)
        _, non_overlap = calculate_overlap(notes_a, stretched_notes_b)

        elapsed_time = time.time() - start_time_global
        eta = (elapsed_time / iteration_count) * (total_iterations - iteration_count)

        print(f"\rProgress: {iteration_count}/{total_iterations} " +
              f"({iteration_count / total_iterations * 100:.2f}%) " +
              f"Current Loss: {non_overlap:.2f} " +
              f"ETA: {eta:.2f}s", end="", flush=True)

        return stretch_factor

    mid_time = (start_time + end_time) / 2
    left_stretch = align_midi_recursive(notes_a, notes_b, start_time, mid_time, min_subdivision, depth + 1)
    right_stretch = align_midi_recursive(notes_a, notes_b, mid_time, end_time, min_subdivision, depth + 1)

    return [left_stretch, right_stretch]


def apply_stretching(notes, stretches, start_time, end_time, min_subdivision=4):
    if isinstance(stretches, float):
        return stretch_notes(notes, start_time, end_time, stretches)

    mid_time = (start_time + end_time) / 2
    left_notes = apply_stretching(notes, stretches[0], start_time, mid_time, min_subdivision)
    right_notes = apply_stretching(notes, stretches[1], mid_time, end_time, min_subdivision)

    return left_notes + right_notes


def stretch_audio(audio_path, stretches, start_time, end_time, min_subdivision=4):
    y, sr = librosa.load(audio_path)

    if isinstance(stretches, float):
        return pyrb.time_stretch(y[int(start_time * sr):int(end_time * sr)], sr, stretches)

    mid_time = (start_time + end_time) / 2
    left_audio = stretch_audio(audio_path, stretches[0], start_time, mid_time, min_subdivision)
    right_audio = stretch_audio(audio_path, stretches[1], mid_time, end_time, min_subdivision)

    return np.concatenate((left_audio, right_audio))


def estimate_total_iterations(start_time, end_time, min_subdivision):
    if end_time - start_time <= min_subdivision:
        return 1
    mid_time = (start_time + end_time) / 2
    return 1 + estimate_total_iterations(start_time, mid_time, min_subdivision) + estimate_total_iterations(mid_time,
                                                                                                            end_time,
                                                                                                            min_subdivision)


def main(midi_a_path, midi_b_path, mp3_c_path, output_midi_path, output_mp3_path):
    global iteration_count, total_iterations, start_time_global

    notes_a = load_midi(midi_a_path)
    notes_b = load_midi(midi_b_path)

    end_time = max(max(note[2] for note in notes_a), max(note[2] for note in notes_b))
    min_subdivision = 4

    total_iterations = estimate_total_iterations(0, end_time, min_subdivision)
    iteration_count = 0
    start_time_global = time.time()

    print("Starting alignment process...")
    stretches = align_midi_recursive(notes_a, notes_b, 0, end_time, min_subdivision)
    print("\nAlignment complete!")

    stretched_notes_b = apply_stretching(notes_b, stretches, 0, end_time)

    # Create a new MIDI file with stretched notes
    new_midi = mido.MidiFile()
    track = mido.MidiTrack()
    new_midi.tracks.append(track)

    current_time = 0
    for note in stretched_notes_b:
        # Ensure note timings are non-negative
        note_on_time = max(0, int((note[1] - current_time) * 1000))
        note_duration = max(0, int((note[2] - note[1]) * 1000))

        if note_on_time > 0 or note_duration > 0:
            track.append(mido.Message('note_on', note=note[0], velocity=64, time=note_on_time))
            track.append(mido.Message('note_off', note=note[0], velocity=64, time=note_duration))
            current_time = note[2]

    new_midi.save(output_midi_path)
    print(f"Stretched MIDI saved to {output_midi_path}")

    # Stretch the MP3 file
    print("Stretching MP3 file...")
    y, sr = librosa.load(mp3_c_path)
    stretched_audio = stretch_audio(mp3_c_path, stretches, 0, end_time)
    sf.write(output_mp3_path, stretched_audio, sr)
    print(f"Stretched MP3 saved to {output_mp3_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align and stretch MIDI and MP3 files.')
    parser.add_argument('midi_a', help='Path to the first MIDI file (A)')
    parser.add_argument('midi_b', help='Path to the second MIDI file (B)')
    parser.add_argument('mp3_c', help='Path to the MP3 file (C)')
    parser.add_argument('output_midi', help='Path for the output stretched MIDI file')
    parser.add_argument('output_mp3', help='Path for the output stretched MP3 file')

    args = parser.parse_args()

    main(args.midi_a, args.midi_b, args.mp3_c, args.output_midi, args.output_mp3)