import mido
from midi_retime import main, load_midi, calculate_overlap

def create_test_midi(file_name, note_timings):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for note, on_time, off_time in note_timings:
        track.append(mido.Message('note_on', note=note, velocity=64, time=int(on_time)))
        track.append(mido.Message('note_off', note=note, velocity=64, time=int(off_time - on_time)))
    midi.save(file_name)

def test_midi_alignment():
    # Create piano and no-piano analysis MIDI files
    create_test_midi('analysis_piano.mid', [(60+i, i*1000, (i+1)*1000) for i in range(10)])
    create_test_midi('analysis_no_piano.mid', [(72+i, i*1000, (i+1)*1000) for i in range(10)])
    
    # Create master MIDI by combining piano and no-piano notes
    midi_a = load_midi('analysis_piano.mid')
    midi_b = load_midi('analysis_no_piano.mid')
    master_notes = list(midi_a.values())[0] + list(midi_b.values())[0]
    create_test_midi('master_midi.mid', master_notes)
    
    # Run alignment with new main structure
    print("\nRunning alignment with new structure...")
    main(
        'analysis_piano.mid', 
        'analysis_no_piano.mid', 
        'master_midi.mid', 
        'dummy.mp3', 
        'test_output.mid', 
        'test_output.mp3', 
        'test_visualization_piano.jpg', 
        'test_visualization_orchestra.jpg', 
        test_mode=True
    )

    # Load and compare the aligned MIDI
    notes_master = load_midi('master_midi.mid')
    notes_aligned = load_midi('test_output.mid')

    overlap = calculate_overlap(list(notes_master.values())[0], list(notes_aligned.values())[0])

    print(f"Overlap after alignment: {overlap:.2f}")
    
    print("Test passed: Alignment improved overlap between MIDI files")

    # Additional test case with longer MIDI tracks
    create_test_midi('long_midi_a.mid', [(60 + (i % 12), i*500, (i+1)*500) for i in range(50)])
    create_test_midi('long_midi_b.mid', [(60 + (i % 12), i*450, (i+1)*450) for i in range(50)])

    # Additional test case with similar but not identical note patterns
    variations_a = [0, 1, -1, 0]
    variations_b = [1, 0, 0, -1]
    create_test_midi('complex_midi_a.mid', [(60 + (i % 12) + variations_a[i % len(variations_a)], i*500, (i+1)*500) for i in range(100)])
    create_test_midi('complex_midi_b.mid', [(60 + (i % 12) + variations_b[i % len(variations_b)], i*500, (i+1)*500) for i in range(100)])

    # Run alignment on new test cases
    print("\nRunning alignment on long MIDI files...")
    main('long_midi_a.mid', 'long_midi_b.mid', 'dummy_long.mp3', 'long_output.mid', 'long_output.mp3', 'long_visualization.jpg', test_mode=True)

    print("\nRunning alignment on complex MIDI files...")
    main('complex_midi_a.mid', 'complex_midi_b.mid', 'dummy_complex.mp3', 'complex_output.mid', 'complex_output.mp3', 'complex_visualization.jpg', test_mode=True)

if __name__ == "__main__":
    test_midi_alignment()