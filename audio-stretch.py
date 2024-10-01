import argparse
import json
import numpy as np
import soundfile as sf
import librosa
from scipy.interpolate import interp1d

def load_timing_dict(piano_json, orchestra_json):
    with open(piano_json, 'r') as f:
        piano_dict = json.load(f)
    with open(orchestra_json, 'r') as f:
        orchestra_dict = json.load(f)
    return piano_dict, orchestra_dict

def combine_timing_dicts(dict1, dict2):
    combined = {}
    keys = sorted(set(dict1.keys()).union(dict2.keys()), key=lambda x: float(x))
    for key in keys:
        speed1 = float(dict1.get(str(key), 1.0))
        speed2 = float(dict2.get(str(key), 1.0))
        combined[float(key)] = (speed1 + speed2) / 2
    return combined

def interpolate_timing(combined_dict, total_duration):
    keys = sorted(combined_dict.keys())
    times = np.array(keys)
    speeds = np.array([combined_dict[k] for k in keys])
    interp_func = interp1d(times, speeds, kind='linear', fill_value="extrapolate")
    return interp_func

def stretch_audio(y, sr, stretches):
    stretched_audio = []
    for start, end, factor in stretches:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        # Ensure factor is within a reasonable range to prevent extreme stretching
        factor = max(0.5, min(factor, 2.0))
        if factor == 1.0:
            stretched_segment = segment
        else:
            stretched_segment = librosa.effects.time_stretch(segment, rate=factor)
        stretched_audio.append(stretched_segment)
    return np.concatenate(stretched_audio)

def main(mp3_path, piano_json, orchestra_json, output_wav):
    piano_dict, orchestra_dict = load_timing_dict(piano_json, orchestra_json)
    combined_dict = combine_timing_dicts(piano_dict, orchestra_dict)
    
    y, sr = librosa.load(mp3_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    interp_func = interpolate_timing(combined_dict, total_duration)
    
    # Create stretches list with overlapping segments to ensure smooth transitions
    stretches = []
    step = 0.1  # 100ms segments
    window = 0.2  # 200ms window for overlap
    for i in np.arange(0, total_duration, step):
        start = i
        end = min(i + window, total_duration)
        speed = interp_func(i)
        factor = speed
        stretches.append((start, end, factor))
    
    warped_audio = stretch_audio(y, sr, stretches)
    
    sf.write(output_wav, warped_audio, sr)
    print(f"Retimed audio saved to {output_wav}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retime MP3 based on timing dictionaries.')
    parser.add_argument('mp3_file', help='Path to the input MP3 file')
    parser.add_argument('piano_json', help='Path to the piano timing JSON file')
    parser.add_argument('orchestra_json', help='Path to the orchestra timing JSON file')
    parser.add_argument('output_wav', help='Path to the output WAV file')
    
    args = parser.parse_args()
    main(args.mp3_file, args.piano_json, args.orchestra_json, args.output_wav)