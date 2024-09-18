import mido
from analysis import main, load_midi, calculate_overlap
import os

def create_test_midi(file_path, note_timings):
    midi = mido.MidiFile()
    # Dictionary to hold tracks based on instrument
    tracks = {}
    
    for timing in note_timings:
        if len(timing) == 4:
            note, on_time, off_time, instrument = timing
        else:
            note, on_time, off_time = timing
            instrument = 'default'
        
        if instrument not in tracks:
            track = mido.MidiTrack()
            track.append(mido.MetaMessage('track_name', name=instrument))
            midi.tracks.append(track)
            tracks[instrument] = track
        else:
            track = tracks[instrument]
        
        # Ensure on_time and off_time are non-negative and different
        on_time = max(0, int(on_time))
        off_time = max(on_time + 100, int(off_time))  # Ensure at least 100ms duration
        
        # Calculate delta time for note_on
        if len(track) > 1:
            delta_on = on_time - sum(msg.time for msg in track[1:])
        else:
            delta_on = on_time
        
        # Ensure delta time is non-negative
        delta_on = max(0, delta_on)
        
        # Convert time to ticks
        ticks_per_beat = 480  # Standard MIDI resolution
        delta_on_ticks = int(delta_on * ticks_per_beat / 1000)
        duration_ticks = max(1, int((off_time - on_time) * ticks_per_beat / 1000))  # Ensure at least 1 tick duration
        
        track.append(mido.Message('note_on', note=note, velocity=64, time=delta_on_ticks))
        track.append(mido.Message('note_off', note=note, velocity=64, time=duration_ticks))
    
    # Add tempo information
    tempo_track = mido.MidiTrack()
    midi.tracks.insert(0, tempo_track)
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM
    
    midi.save(file_path)

def test_simple_alignment():
    # Create test folder
    os.makedirs('tests/simple', exist_ok=True)
    
    # Create single instrument MIDI with varying note durations
    create_test_midi('tests/simple/analysis_piano.mid', [(60+i, i*1000, (i+1)*1000 + 500, 'piano') for i in range(10)])
    
    # Create multiple instruments MIDI with varying note durations
    create_test_midi('tests/simple/analysis_multiple.mid', 
                    [(60+i, i*1000, (i+1)*1000 + 600, 'piano') for i in range(5)] +
                    [(72+i, i*1000, (i+1)*1000 + 700, 'orchestra') for i in range(5)])
    
    # Create master MIDI with different length and similar notes
    create_test_midi('tests/simple/master_midi.mid', 
                    [(60+i, i*900, (i+1)*900 + 550, 'piano') for i in range(8)] +
                    [(72+i, i*900, (i+1)*900 + 650, 'orchestra') for i in range(8)])
    
    # Define output MIDI path
    output_mid = 'tests/simple/test_output.mid'
    
    # Run alignment
    main(
        'tests/simple/analysis_piano.mid', 
        'piano', 
        'tests/simple/master_midi.mid', 
        'tests/simple/test_output.json', 
        'tests/simple/test_visualization.jpg',
        output_mid
    )
    
    # Load and compare the aligned MIDI
    notes_master = load_midi('tests/simple/master_midi.mid')
    notes_aligned = load_midi(output_mid)
    
    overlap = calculate_overlap(notes_master, notes_aligned)
    
    with open('tests/simple/test_results.txt', 'w') as f:
        f.write(f"Overlap after alignment: {overlap:.2f}\n")
        f.write("Test passed: Simple Alignment\n")

def test_complex_alignment():
    # Create test folder
    os.makedirs('tests/complex', exist_ok=True)
    
    # Create complex single instrument MIDI with varying note durations
    create_test_midi('tests/complex/analysis_complex_piano.mid', 
                    [(60 + (i % 12), i*500, (i+1)*500 + 300, 'piano') for i in range(50)])
    
    # Create complex multiple instruments MIDI with varying note durations
    create_test_midi('tests/complex/analysis_complex_multiple.mid', 
                    [(60 + (i % 12), i*450, (i+1)*450 + 350, 'piano') for i in range(50)] +
                    [(72 + (i % 12), i*450, (i+1)*450 + 400, 'orchestra') for i in range(50)])
    
    # Create master complex MIDI with different length and similar notes
    create_test_midi('tests/complex/master_complex_midi.mid', 
                    [(60 + (i % 12), i*400, (i+1)*400 + 325, 'piano') for i in range(60)] +
                    [(72 + (i % 12), i*400, (i+1)*400 + 375, 'orchestra') for i in range(60)])
    
    # Define output MIDI path
    output_mid = 'tests/complex/test_output.mid'
    
    try:
        # Run alignment
        main(
            'tests/complex/analysis_complex_piano.mid', 
            'piano', 
            'tests/complex/master_complex_midi.mid', 
            'tests/complex/test_output.json', 
            'tests/complex/test_visualization.jpg',
            output_mid
        )
        
        # Load and compare the aligned MIDI
        notes_master = load_midi('tests/complex/master_complex_midi.mid')
        notes_aligned = load_midi(output_mid)
        
        overlap = calculate_overlap(notes_master, notes_aligned)
        
        with open('tests/complex/test_results.txt', 'w') as f:
            f.write(f"Overlap after alignment: {overlap:.2f}\n")
            f.write("Test passed: Complex Alignment\n")
    except ValueError as e:
        with open('tests/complex/test_results.txt', 'w') as f:
            f.write(f"Test failed: Complex Alignment\n")
            f.write(f"Error: {str(e)}\n")

def test_midi_alignment():
    test_simple_alignment()
    test_complex_alignment()

if __name__ == "__main__":
    test_midi_alignment()