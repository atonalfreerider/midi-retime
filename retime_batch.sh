#!/usr/bin/env bash
set -euo pipefail

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

  echo "\nProcessing timing file: $timing"

  # find audio file in same dir (prefer no_piano variants)
  audio_file=""
  # search for no_piano variants first
  audio_file_candidate=$(ls "$timing_dir"/*no*piano*.wav 2>/dev/null | head -n1 || true)
  if [[ -n "$audio_file_candidate" ]]; then
    audio_file="$audio_file_candidate"
  else
    audio_file_candidate=$(ls "$timing_dir"/*.wav 2>/dev/null | head -n1 || true)
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
  # create tokens (words >=4 chars) from timing_base
  read -ra TOKENS <<< "$(echo "$timing_base" | sed 's/[^a-zA-Z0-9]/ /g')"

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

  if [[ -z "$midi_match" ]]; then
    echo "No MIDI match found for timing $timing_base under $MUSESCORE_DIR. Skipping."
    continue
  fi

  echo "Matched MIDI: $midi_match"

  # Build output paths
  out_json="$RETIMES_DIR/${timing_slug}-retime.json"
  out_wav="$RETIMES_DIR/${timing_slug}-retime.wav"

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
