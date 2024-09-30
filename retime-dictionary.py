import argparse
import json
import mido
from typing import Dict, Tuple
from scipy import interpolate

def parse_timing_dict(timing_file: str) -> Dict[int, float]:
    timing_dict = {}
    with open(timing_file, 'r') as f:
        for line in f:
            measure, time_str = line.strip().split('|')
            measure = int(measure)
            minutes, seconds = time_str.split(':')
            seconds, milliseconds = seconds.split('.')
            time_in_seconds = (int(minutes) * 60) + int(seconds) + (int(milliseconds) / 1000)
            timing_dict[measure] = time_in_seconds
    return timing_dict


def get_midi_timings(midi_file: str) -> Dict[int, float]:
    midi = mido.MidiFile(midi_file)
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # Default tempo (microseconds per beat)
    time_signature = (4, 4)  # Default time signature
    ticks_per_measure = ticks_per_beat * time_signature[0]

    # Collect all MIDI events
    events = []
    for track in midi.tracks:
        absolute_tick = 0
        for msg in track:
            absolute_tick += msg.time
            events.append((absolute_tick, msg))

    # Sort events by their absolute tick count
    events.sort(key=lambda x: x[0])

    midi_timings = {1: 0.0}  # Measure 1 starts at t=0
    current_measure = 1
    current_time = 0.0
    current_ticks = 0
    measure_start_ticks = 0
    next_measure_ticks = ticks_per_measure

    for event_ticks, msg in events:
        # Calculate time up to this event
        delta_ticks = event_ticks - current_ticks
        current_time += (delta_ticks * tempo) / (1000000 * ticks_per_beat)
        current_ticks = event_ticks

        # Check if we've reached or passed measure boundaries
        while current_ticks >= next_measure_ticks:
            current_measure += 1
            measure_start_time = current_time - ((current_ticks - next_measure_ticks) * tempo) / (1000000 * ticks_per_beat)
            midi_timings[current_measure] = measure_start_time
            measure_start_ticks = next_measure_ticks
            next_measure_ticks += ticks_per_measure

        # Process tempo and time signature changes
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'time_signature':
            # Update time signature and ticks per measure
            ticks_per_measure = ticks_per_beat * 4 * msg.numerator // msg.denominator
            
            # Adjust the next measure boundary
            next_measure_ticks = measure_start_ticks + ticks_per_measure

    # Process any remaining ticks after the last event
    while current_ticks >= next_measure_ticks:
        current_measure += 1
        measure_start_time = current_time + ((next_measure_ticks - current_ticks) * tempo) / (1000000 * ticks_per_beat)
        midi_timings[current_measure] = measure_start_time
        next_measure_ticks += ticks_per_measure

    return midi_timings


def create_stretching_map(wav_timings: Dict[int, float], midi_timings: Dict[int, float]) -> Dict[float, Tuple[float, float]]:
    stretching_map = {}
    wav_measures = sorted(wav_timings.keys())
    midi_measures = sorted(midi_timings.keys())
    
    for wav_measure in wav_measures:
        wav_time = wav_timings[wav_measure]
        
        # Find the corresponding MIDI measure
        midi_measure = next((m for m in midi_measures if m >= wav_measure), None)
        
        if midi_measure is None:
            print(f"Warning: Could not find corresponding MIDI measure for WAV measure {wav_measure}")
            continue
        
        midi_time = midi_timings[midi_measure]
        
        stretching_map[wav_time] = (midi_time, 1)  # We'll calculate stretch factors in retime_audio
    
    return stretching_map


def create_timing_graph(midi_timings: Dict[int, float], wav_timings: Dict[int, float], output_jpg: str):
    plt.figure(figsize=(12, 8))
    
    # Set up the plot
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_ylim(max(max(midi_timings.values()), max(wav_timings.values())) * 1.1, 0)
    
    # Plot MIDI timings
    midi_measures = sorted(midi_timings.keys())
    midi_times = [midi_timings[m] for m in midi_measures]
    plt.plot([-0.2] * len(midi_times), midi_times, 'r-', label='MIDI')
    
    # Plot WAV timings
    wav_measures = sorted(wav_timings.keys())
    wav_times = [wav_timings[m] for m in wav_measures]
    plt.plot([1.2] * len(wav_times), wav_times, 'g-', label='WAV')
    
    # Add hash marks and labels only for WAV measures (from txt file)
    for measure in wav_measures:
        midi_time = midi_timings.get(measure, None)
        wav_time = wav_timings[measure]
        
        # MIDI hash mark and label (only if measure exists in MIDI)
        if midi_time is not None:
            plt.plot([-0.25, -0.15], [midi_time, midi_time], 'r-')
            plt.text(-0.3, midi_time, f'M{measure}', ha='right', va='center')
        
        # WAV hash mark and label
        plt.plot([1.15, 1.25], [wav_time, wav_time], 'g-')
        plt.text(1.3, wav_time, f'M{measure}', ha='left', va='center')
        
        # Draw skew line and add ratio label
        if midi_time is not None:
            plt.plot([-0.2, 1.2], [midi_time, wav_time], 'm-', alpha=0.5)
            
            # Calculate and display the ratio
            ratio = midi_time / wav_time if wav_time != 0 else float('inf')
            mid_x = 0.5
            mid_y = (midi_time + wav_time) / 2
            plt.text(mid_x, mid_y, f'{ratio:.2f}', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.title('MIDI vs WAV Timing Comparison')
    plt.legend(loc='lower right')
    plt.axis('off')
    
    # Save the graph
    plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Apply a stretching map to an audio file.")    
    parser.add_argument("midi_file", help="Path to MIDI master timing file")
    parser.add_argument("timing_file", help="Path to timing dictionary text file")
    parser.add_argument("output_json", help="Path to output stretching map json file")
    args = parser.parse_args()

    audio_timings = parse_timing_dict(args.timing_file)
    midi_timings = get_midi_timings(args.midi_file)
    
    if not audio_timings or not midi_timings:
        print("Error: Empty timing information. Please check your input files.")
        return

    stretching_map = create_stretching_map(audio_timings, midi_timings)
        
    with open(args.output_json, 'w') as f:
        json.dump(stretching_map, f, indent=4)


if __name__ == "__main__":
    main()