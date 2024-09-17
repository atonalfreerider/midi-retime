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
    keys = sorted(set(dict1.keys()).union(dict2.keys()))
    for key in keys:
        speed1 = dict1.get(str(key), 1.0)
        speed2 = dict2.get(str(key), 1.0)
        combined[key] = (float(speed1) + float(speed2)) / 2
    return combined

def interpolate_timing(combined_dict, total_duration, sr):
    keys = sorted(combined_dict.keys(), key=lambda x: float(x))
    times = np.array([float(k) for k in keys])
    speeds = np.array([combined_dict[k] for k in keys])
    interp_func = interp1d(times, speeds, kind='linear', fill_value='extrapolate')
    new_times = np.linspace(0, total_duration, int(total_duration * sr))
    interpolated_speeds = interp_func(new_times)
    return new_times, interpolated_speeds

def apply_time_stretch(y, sr, times, speeds):
    stretched_audio = []
    for i in range(len(times)-1):
        start = int(times[i] * sr)
        end = int(times[i+1] * sr)
        segment = y[start:end]
        speed = speeds[i]
        if speed != 1.0 and len(segment) > 1:
            segment = librosa.effects.time_stretch(segment, rate=speed)
        stretched_audio.append(segment)
    return np.concatenate(stretched_audio)

def main(mp3_path, piano_json, orchestra_json, output_wav):
    piano_dict, orchestra_dict = load_timing_dict(piano_json, orchestra_json)
    combined_dict = combine_timing_dicts(piano_dict, orchestra_dict)
    
    y, sr = librosa.load(mp3_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    times, speeds = interpolate_timing(combined_dict, total_duration, sr)
    warped_audio = apply_time_stretch(y, sr, times, speeds)
    
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