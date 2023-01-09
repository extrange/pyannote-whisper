# Speaker Diarization with Pyannote and [Whisper.cpp][whisper.cpp]

Uses [Whisper.cpp][whisper.cpp] to transcribe audio, and then performs speaker diarization with [Pyannote][pyannote].

## Usage

Place video/audio files in `input/`, and then run `main.py` with `docker compose up`.

## Notes

Performance for diarization seems to be improved when segment length for `whisper` is decreased, such as `--max-len 50`.

[whisper.cpp]: https://github.com/ggerganov/whisper.cpp
[pyannote]: https://github.com/pyannote/pyannote-audio