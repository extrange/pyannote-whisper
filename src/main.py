from typing import NamedTuple
import sys

from pyannote.audio import Pipeline
from dotenv import dotenv_values
import subprocess
from pathlib import Path

HF_TOKEN = dotenv_values(Path(__file__).parent / ".env")["HF_TOKEN"] or ""
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"


class TranscribedSegment(NamedTuple):
    start: int
    end: int
    text: str


class SpeakerSegment(NamedTuple):
    start: int
    end: int
    speaker: str


def millisec(timeStr):
    """Convert timestring in the form HH:MM:SS.MMM to milliseconds"""
    spl = timeStr.split(":")
    s = int((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


def overlap(text: TranscribedSegment, diarization: SpeakerSegment):
    """Calculate overlap between segments in ms"""
    return max(0, min(text.end, diarization.end) - max(text.start, diarization.start))


def main():
    # Process all files in input/
    for f in INPUT_DIR.iterdir():
        process(f)


def process(file_to_process: Path):
    # Convert with ffmpeg
    wav_path = OUTPUT_DIR / (file_to_process.stem + ".wav")
    print(f"Converting {file_to_process} to .wav...")
    subprocess.run(
        f"ffmpeg -y -i '{file_to_process}' -ar 16000 -ac 1 -c:a pcm_s16le '{wav_path}'",
        shell=True,
        stderr=subprocess.STDOUT,
        stdout=sys.stdout,
    )

    # Transcribe and save
    print(f"Transcribing with Whisper...")
    subprocess.run(
        f"/whisper.cpp/main -m /models/ggml-large.bin -ml 50 -ocsv -f '{wav_path}'",
        shell=True,
        stderr=subprocess.STDOUT,
        stdout=sys.stdout,
    )
    transcript_path = wav_path.parent / f"{wav_path.name}.csv"

    # Diarize and save
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN,
    )
    diarization = pipeline(wav_path)
    diarization_file = OUTPUT_DIR / (file_to_process.name + "-diarization.txt")
    with diarization_file.open("w") as f:
        f.write(str(diarization))

    speaker_segments = [
        SpeakerSegment(int(track[0].start * 1000), int(track[0].end * 1000), track[2])
        for track in diarization.itertracks(yield_label=True)
    ]

    # Read whisper transcript
    with open(transcript_path, newline="") as f:
        lines = [line.split(" ", 2) for line in f.readlines()]

        transcribed_segments = [
            TranscribedSegment(int(line[0][:-1]), int(line[1][:-1]), line[2][1:-2])
            for line in lines
        ]

    # Match transcribed segments to the speaker segment
    # Based on highest fraction of overlap
    matched = []
    for transcribed_segment in transcribed_segments:
        speaker = max(
            speaker_segments,
            key=lambda s: overlap(transcribed_segment, s) / (s.end - s.start),
        ).speaker
        matched.append([speaker, transcribed_segment.text])

    # Merge lines where the speaker is unchanged
    prev_speaker = None
    matched_merged = []
    for match in matched:
        current_speaker = match[0]
        if current_speaker == prev_speaker:
            # If same speaker, join with previous + space
            matched_merged[-1][1] += f" {match[1]}"
        else:
            # Otherwise, add as a new line
            matched_merged.append(match)
        prev_speaker = current_speaker

    # Write to output
    output = OUTPUT_DIR / (file_to_process.stem + "-output.txt")
    with output.open("w") as f:
        f.writelines([f"{a[0]}: {a[1]}\n\n" for a in matched_merged])


if __name__ == "__main__":
    main()
