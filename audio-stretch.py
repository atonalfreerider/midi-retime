import argparse
import json
import numpy as np
import soundfile as sf
import librosa
from scipy import interpolate
from typing import Dict, Tuple

def retime_audio(input_audio: str, stretching_map: Dict[str, Tuple[float, float]], output_wav: str):
    # Load the audio file
    y, sr = librosa.load(input_audio, sr=None)
    
    # Create time array
    time = np.arange(len(y)) / sr
    
    # Sort the stretching map by original time
    sorted_stretch_points = sorted(stretching_map.items(), key=lambda x: float(x[0]))
    
    # Separate the original and new time points
    orig_times, new_times = zip(*sorted_stretch_points)
    orig_times = np.array([float(t) for t in orig_times])
    new_times = np.array([nt[0] for nt in new_times])  # Extract just the time value, not the tuple
    
    # Create a piecewise linear interpolation function
    time_map = interpolate.interp1d(orig_times, new_times, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Get the last keypoint times
    last_orig_time = float(sorted_stretch_points[-1][0])
    last_new_time = sorted_stretch_points[-1][1][0]
    
    # Calculate the end time of the tail section
    tail_duration = min(10.0, (len(y) / sr) - last_orig_time)  # Either 10 seconds or remaining audio
    final_orig_time = last_orig_time + tail_duration
    final_new_time = last_new_time + tail_duration  # 1:1 mapping for tail
    
    # Initialize the output audio array
    y_retimed = np.zeros(int(final_new_time * sr))
    
    # Perform the time stretching using a phase vocoder for all segments except the last
    for i in range(len(sorted_stretch_points) - 1):
        start_time, (new_start_time, _) = sorted_stretch_points[i]
        end_time, (new_end_time, _) = sorted_stretch_points[i + 1]
        
        # Convert string times to float
        start_time, end_time = float(start_time), float(end_time)
        
        # Extract the segment
        segment_mask = (time >= start_time) & (time < end_time)
        segment = y[segment_mask]
        
        # Calculate the average stretch factor for this segment
        avg_stretch = (new_end_time - new_start_time) / (end_time - start_time)
        
        # Time-stretch the segment
        stretched_segment = librosa.effects.time_stretch(segment, rate=1/avg_stretch)
        
        # Calculate the new start and end indices
        new_start_idx = int(new_start_time * sr)
        new_end_idx = int(new_end_time * sr)
        
        # Ensure the stretched segment fits exactly in the allocated space
        stretched_segment = librosa.util.fix_length(stretched_segment, size=new_end_idx - new_start_idx)
        
        # Insert the stretched segment into the output array
        y_retimed[new_start_idx:new_end_idx] = stretched_segment
    
    # Handle the tail section (last segment plus up to 10 seconds)
    last_start_idx = int(last_new_time * sr)
    last_end_idx = int(final_new_time * sr)
    
    # Extract the tail section from input audio (no stretching needed)
    tail_start_idx = int(last_orig_time * sr)
    tail_end_idx = int(final_orig_time * sr)
    tail_segment = y[tail_start_idx:tail_end_idx]
    
    # Ensure the tail segment fits exactly
    tail_segment = librosa.util.fix_length(tail_segment, size=last_end_idx - last_start_idx)
    
    # Insert the tail segment into the output array
    y_retimed[last_start_idx:last_end_idx] = tail_segment
    
    # Save the retimed audio
    sf.write(output_wav, y_retimed, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retime audio based on timing scale.')
    parser.add_argument('audio_file', help='Path to the input MP3 or WAV file')
    parser.add_argument('timing_scale', help='Path to the timing scale file')
    parser.add_argument('output_wav', help='Path to the output WAV file')
    
    args = parser.parse_args()

    with open(args.timing_scale, 'r') as f:
        timing_scale = json.load(f)

    retime_audio(args.audio_file, timing_scale, args.output_wav)
    print(f"Retimed audio saved to {args.output_wav}")