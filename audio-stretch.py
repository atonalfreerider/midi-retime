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
    
    # Apply the time mapping to all sample times
    new_time = time_map(time)
    
    # Calculate the instantaneous stretch factor for each sample
    stretch_factors = np.diff(new_time) / np.diff(time)
    stretch_factors = np.insert(stretch_factors, 0, stretch_factors[0])  # Pad the first element
    
    # Initialize the output audio array
    y_retimed = np.zeros(int(new_time[-1] * sr))
    
    # Perform the time stretching using a phase vocoder
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
    
    # Handle the last segment
    last_start_time, (last_new_start_time, _) = sorted_stretch_points[-1]
    last_start_time = float(last_start_time)
    last_segment = y[time >= last_start_time]
    last_stretch = (new_time[-1] - last_new_start_time) / (time[-1] - last_start_time)
    stretched_last_segment = librosa.effects.time_stretch(last_segment, rate=1/last_stretch)
    
    last_start_idx = int(last_new_start_time * sr)
    y_retimed[last_start_idx:] = librosa.util.fix_length(stretched_last_segment, size=len(y_retimed) - last_start_idx)
    
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