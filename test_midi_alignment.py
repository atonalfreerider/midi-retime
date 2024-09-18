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
        
        track.append(mido.Message('note_on', note=note, velocity=64, time=int(on_time)))
        track.append(mido.Message('note_off', note=note, velocity=64, time=int(off_time - on_time)))
    
    midi.save(file_path)

def test_simple_alignment():
    # Create test folder
    os.makedirs('tests/simple', exist_ok=True)
    
    # Create single instrument MIDI
    create_test_midi('tests/simple/analysis_piano.mid', [(60+i, i*1000, (i+1)*1000, 'piano') for i in range(10)])
    
    # Create multiple instruments MIDI
    create_test_midi('tests/simple/analysis_multiple.mid', 
                    [(60+i, i*1000, (i+1)*1000, 'piano') for i in range(5)] +
                    [(72+i, i*1000, (i+1)*1000, 'orchestra') for i in range(5)])
    
    # Create master MIDI with different length and similar notes
    create_test_midi('tests/simple/master_midi.mid', 
                    [(60+i, i*900, (i+1)*900, 'piano') for i in range(8)] +
                    [(72+i, i*900, (i+1)*900, 'orchestra') for i in range(8)])
    
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
    
    # Check if 'piano' key exists in both dictionaries
    if 'piano' in notes_master and 'piano' in notes_aligned:
        overlap = calculate_overlap(notes_master['piano'], notes_aligned['piano'])
        
        with open('tests/simple/test_results.txt', 'w') as f:
            f.write(f"Overlap after alignment: {overlap:.2f}\n")
            f.write("Test passed: Simple Alignment\n")
    else:
        with open('tests/simple/test_results.txt', 'w') as f:
            f.write("Error: 'piano' track not found in one or both MIDI files\n")
            f.write("Test failed: Simple Alignment\n")

def test_complex_alignment():
    # Create test folder
    os.makedirs('tests/complex', exist_ok=True)
    
    # Create complex single instrument MIDI
    create_test_midi('tests/complex/analysis_complex_piano.mid', 
                    [(60 + (i % 12), i*500, (i+1)*500, 'piano') for i in range(50)])
    
    # Create complex multiple instruments MIDI
    create_test_midi('tests/complex/analysis_complex_multiple.mid', 
                    [(60 + (i % 12), i*450, (i+1)*450, 'piano') for i in range(50)] +
                    [(72 + (i % 12), i*450, (i+1)*450, 'orchestra') for i in range(50)])
    
    # Create master complex MIDI with different length and similar notes
    create_test_midi('tests/complex/master_complex_midi.mid', 
                    [(60 + (i % 12), i*400, (i+1)*400, 'piano') for i in range(60)] +
                    [(72 + (i % 12), i*400, (i+1)*400, 'orchestra') for i in range(60)])
    
    # Define output MIDI path
    output_mid = 'tests/complex/test_output.mid'
    
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
    
    # Check if 'piano' key exists in both dictionaries
    if 'piano' in notes_master and 'piano' in notes_aligned:
        overlap = calculate_overlap(notes_master['piano'], notes_aligned['piano'])
        
        with open('tests/complex/test_results.txt', 'w') as f:
            f.write(f"Overlap after alignment: {overlap:.2f}\n")
            f.write("Test passed: Complex Alignment\n")
    else:
        with open('tests/complex/test_results.txt', 'w') as f:
            f.write("Error: 'piano' track not found in one or both MIDI files\n")
            f.write("Test failed: Complex Alignment\n")

def test_midi_alignment():
    test_simple_alignment()
    test_complex_alignment()

if __name__ == "__main__":
    test_midi_alignment()