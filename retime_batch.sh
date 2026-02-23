#!/usr/bin/env bash
set -eu

usage() {
  cat <<EOF
Usage: $0 --env CONDA_ENV --music-dir MUSIC_DIR --musescore-dir MUSESCORE_DIR [--retimes-dir OUTDIR] [--dry-run]

Scans MUSIC_DIR for "measure-timing" files, finds matching MIDI files under MUSESCORE_DIR,
and runs retime-dictionary.py and audio-stretch.py for each match.

Options:
  --env           Conda environment name to activate (required)
  --music-dir     Path to your Music folder (where measure-timing.txt and audio live)
  --musescore-dir Path to your MuseScore folder (where .mid files live)
  --retimes-dir   Output directory for retime JSON and retimed WAVs (default: ./retimes)
  --dry-run       Show what would be executed, don't run commands
  -h, --help      Show this help
EOF
}

CONDA_ENV=""
MUSIC_DIR=""
MUSESCORE_DIR=""
RETIMES_DIR="$(pwd)/retimes"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) CONDA_ENV="$2"; shift 2;;
    --music-dir) MUSIC_DIR="$2"; shift 2;;
    --musescore-dir) MUSESCORE_DIR="$2"; shift 2;;
    --retimes-dir) RETIMES_DIR="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$CONDA_ENV" || -z "$MUSIC_DIR" || -z "$MUSESCORE_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

mkdir -p "$RETIMES_DIR"

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    # Initialize conda for this shell
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  else
    # Fallback: try to source a common conda location
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
    else
      echo "conda not found and fallback not available. Please ensure conda is installed." >&2
      exit 2
    fi
  fi
}

run_cmd() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    eval "$@"
  fi
}

# Activate conda
echo "Activating conda environment: $CONDA_ENV"
run_cmd "activate_conda"

# Find all measure-timing files under MUSIC_DIR
mapfile -t TIMING_FILES < <(find "$MUSIC_DIR" -type f -iname '*measure-timing.txt' 2>/dev/null)

if [[ ${#TIMING_FILES[@]} -eq 0 ]]; then
  echo "No measure-timing.txt files found under $MUSIC_DIR"
  exit 0
fi

for timing in "${TIMING_FILES[@]}"; do
  timing_dir=$(dirname "$timing")
  timing_base=$(basename "$timing" .txt)
  timing_key=$(echo "$timing_base" | sed 's/[- ]/ /g')
  timing_slug=$(echo "$timing_base" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')

  echo ""
  echo "Processing timing file: $timing"
  echo "  Timing base: $timing_base"

  # find audio file in same dir (prefer no_piano variants, support mp3/wav/flac)
  audio_file=""
  # search for no_piano variants first (any audio format)
  audio_file_candidate=$(find "$timing_dir" -maxdepth 1 -iname '*no*piano*.mp3' -o -iname '*no*piano*.wav' -o -iname '*no*piano*.flac' 2>/dev/null | head -n1 || true)
  if [[ -n "$audio_file_candidate" ]]; then
    audio_file="$audio_file_candidate"
  else
    # fallback: find any audio file
    audio_file_candidate=$(find "$timing_dir" -maxdepth 1 \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' \) 2>/dev/null | head -n1 || true)
    if [[ -n "$audio_file_candidate" ]]; then
      audio_file="$audio_file_candidate"
    fi
  fi

  if [[ -z "$audio_file" ]]; then
    echo "Warning: No WAV file found in $timing_dir for timing $timing_base. Audio-stretch will be skipped for this timing."
  else
    echo "Found audio: $audio_file"
  fi

  # Attempt to match MIDI file in MUSESCORE_DIR
  midi_match=""
  
  # Extract composition number and movement from timing_base more carefully
  # e.g., "Rach2-3-Bronfman" -> comp_num="2", movement="3"
  # e.g., "Ohlsson-Rach3-1-measure" -> comp_num="3", movement="1"
  comp_num=$(echo "$timing_base" | sed -E 's/.*[Rr]ach([23]).*/\1/')
  movement=$(echo "$timing_base" | sed -E 's/.*[Rr]ach[23]-([123]).*/\1/')
  
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  comp_num: '$comp_num', movement: '$movement'"
  fi
  
  # Create tokens from timing_base (words >=4 chars)
  read -ra TOKENS <<< "$(echo "$timing_base" | sed 's/[^a-zA-Z0-9]/ /g')"
  
  # First pass: look for master.mid files with composition key, and match movement by folder
  if [[ -n "$comp_num" ]]; then
    # Build the movement folder name (1->I, 2->II, 3->III)
    movement_folder=""
    case "$movement" in
      1) movement_folder="I" ;;
      2) movement_folder="II" ;;
      3) movement_folder="III" ;;
    esac
    
    if [[ -n "$movement_folder" ]]; then
      # Try to find any .mid file in the specific movement folder (prefer master.mid, but accept any .mid)
      # Search for Rach-2 or Rach-3 folder
      comp_dir=$(find "$MUSESCORE_DIR" -type d -iname "Rach-$comp_num" 2>/dev/null | head -1)
      if [[ -n "$comp_dir" ]]; then
        # First try master.mid
        midi_match=$(ls -1 "$comp_dir/$movement_folder"/*master.mid 2>/dev/null | head -1)
        # If no master.mid, accept any .mid file in that movement folder
        if [[ -z "$midi_match" ]]; then
          midi_match=$(ls -1 "$comp_dir/$movement_folder"/*.mid 2>/dev/null | head -1)
        fi
      fi
    fi
    
    # If no movement folder match, try any master.mid in the composition folder
    if [[ -z "$midi_match" ]]; then
      comp_dir=$(find "$MUSESCORE_DIR" -type d -iname "Rach-$comp_num" 2>/dev/null | head -1)
      if [[ -n "$comp_dir" ]]; then
        midi_match=$(find "$comp_dir" -name "*master.mid" 2>/dev/null | head -1)
      fi
    fi
  fi
  
  # Second pass: token-based matching on all MIDI files (fallback)
  if [[ -z "$midi_match" ]]; then
    while IFS= read -r midi; do
      midi_base=$(basename "$midi")
      midi_base_lc=$(echo "$midi_base" | tr '[:upper:]' '[:lower:]')
      for token in "${TOKENS[@]}"; do
        token_lc=$(echo "$token" | tr '[:upper:]' '[:lower:]')
        if [[ ${#token_lc} -ge 4 && "$midi_base_lc" == *"$token_lc"* ]]; then
          midi_match="$midi"
          break 2
        fi
      done
    done < <(find "$MUSESCORE_DIR" -type f \( -iname '*.mid' -o -iname '*.midi' \) 2>/dev/null)
  fi

  if [[ -z "$midi_match" ]]; then
    echo "No MIDI match found for timing $timing_base under $MUSESCORE_DIR. Skipping."
    continue
  fi

  echo "Matched MIDI: $midi_match"

  # Build output paths
  out_json="$RETIMES_DIR/${timing_slug}-retime.json"
  out_wav="$RETIMES_DIR/${timing_slug}-retime.wav"

  # Skip if output WAV exists and MIDI is not significantly newer (within 5 minutes)
  if [[ -f "$out_wav" ]]; then
    midi_mtime=$(stat -c %Y "$midi_match" 2>/dev/null || echo 0)
    wav_mtime=$(stat -c %Y "$out_wav" 2>/dev/null || echo 0)
    age_diff=$((midi_mtime - wav_mtime))
    
    if [[ $age_diff -lt 300 ]]; then
      echo "Output WAV already exists and MIDI is not significantly newer. Skipping."
      continue
    fi
  fi

  # Run retime-dictionary.py
  cmd1="python3 ./retime-dictionary.py \"$midi_match\" \"$timing\" \"$out_json\""
  run_cmd "$cmd1"

  # Run audio-stretch.py if audio file found
  if [[ -n "$audio_file" ]]; then
    cmd2="python3 ./audio-stretch.py \"$audio_file\" \"$out_json\" \"$out_wav\""
    run_cmd "$cmd2"
  fi

done

echo "\nBatch processing complete. Outputs (JSON/WAV) are in: $RETIMES_DIR"
