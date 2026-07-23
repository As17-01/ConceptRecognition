import re
import subprocess
import sys
import tempfile

import hydra

from pathlib import Path
from omegaconf import DictConfig
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings

# Structural pause only - splitting on micro-pauses too made ~380 independent TTS calls per class,
# which audibly drifted in timbre/prosody between segments. Micro-pause markers within a block are
# stripped for synthesis (speech runs continuously until the next [ПАУЗА:N]).
STRUCTURAL_PAUSE_RE = re.compile(r"\[ПАУЗА:(\d+)\]")
MICRO_PAUSE_RE = re.compile(r"\[МИКРОПАУЗА:\d+\]")
BRACKETED_MARKER_RE = re.compile(r"\[(?P<kind>[^\]:]+):(?P<seconds>\d+)\]")


def normalize_pause_markers(text: str) -> tuple[str, int]:
    """Fold LLM/transcription typos (e.g. [MIKROPAUZA:N], [MIKРОПАУЗА:N]) into canonical markers."""
    fixes = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal fixes
        kind = match.group("kind")
        seconds = match.group("seconds")
        cyr = re.sub(r"[^А-ЯЁ]", "", kind.upper())
        ascii_letters = re.sub(r"[^A-Za-z]", "", kind).upper()

        if cyr == "ПАУЗА":
            canonical = f"[ПАУЗА:{seconds}]"
        elif cyr == "МИКРОПАУЗА" or (
            "MIK" in ascii_letters and ("PAUZ" in ascii_letters or "ПАУЗ" in cyr)
        ):
            canonical = f"[МИКРОПАУЗА:{seconds}]"
        else:
            return match.group(0)

        if canonical != match.group(0):
            fixes += 1
        return canonical

    return BRACKETED_MARKER_RE.sub(repl, text), fixes


def split_structural_blocks(text: str) -> list[tuple[str, int]]:
    """Splits on [ПАУЗА:N] only into (spoken_text, structural_pause_seconds_after) pairs."""
    parts = STRUCTURAL_PAUSE_RE.split(text)
    blocks = []
    for i in range(0, len(parts), 2):
        spoken = parts[i].strip()
        pause_seconds = int(parts[i + 1]) if i + 1 < len(parts) else 0
        blocks.append((spoken, pause_seconds))
    return blocks


def text_for_tts(spoken: str) -> str:
    """Strips micro-pause markers so each structural block is one continuous TTS utterance."""
    return re.sub(r"\s+", " ", MICRO_PAUSE_RE.sub(" ", spoken)).strip()


def context_snippet(text: str, max_chars: int, from_end: bool) -> str | None:
    if not text or max_chars <= 0:
        return None
    snippet = text[-max_chars:] if from_end else text[:max_chars]
    return snippet or None


def synthesize_segment(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    voice_settings: VoiceSettings,
    language_code: str | None,
    previous_text: str | None = None,
    next_text: str | None = None,
    seed: int | None = None,
) -> bytes:
    kwargs = dict(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
        voice_settings=voice_settings,
        language_code=language_code,
    )
    if previous_text:
        kwargs["previous_text"] = previous_text
    if next_text:
        kwargs["next_text"] = next_text
    if seed is not None:
        kwargs["seed"] = seed

    chunks = client.text_to_speech.convert(**kwargs)
    return b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))


def mp3_bytes_to_wav(audio: bytes, work_dir: Path, name: str) -> Path:
    src_path = work_dir / f"{name}.mp3"
    src_path.write_bytes(audio)
    wav_path = work_dir / f"{name}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path), "-ar", "44100", "-ac", "1", str(wav_path)],
        capture_output=True,
        check=True,
    )
    return wav_path


def silence_wav(seconds: int, work_dir: Path, name: str) -> Path:
    wav_path = work_dir / f"{name}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(seconds),
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )
    return wav_path


def concat_to_mp3(wav_paths: list[Path], dst_path: Path, work_dir: Path, bitrate: str) -> None:
    list_path = work_dir / "concat_list.txt"
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in wav_paths), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-b:a", bitrate, str(dst_path)],
        capture_output=True,
        check=True,
    )


@hydra.main(config_path="../conf", config_name="synthesize_speech", version_base=None)
def main(cfg: DictConfig) -> None:
    src_dir = Path(cfg.src)
    if not src_dir.is_dir():
        print(f"Source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    voice_id_path = Path(cfg.voice_id_file)
    if not voice_id_path.is_file():
        print(f"Voice ID file not found: {voice_id_path} - run clone_voice.py first", file=sys.stderr)
        sys.exit(1)
    voice_id = voice_id_path.read_text(encoding="utf-8").strip()

    txt_files = sorted(src_dir.glob("*.txt"))
    if not txt_files:
        print(f"No generated class scripts found in {src_dir}")
        return

    dst_dir = Path(cfg.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    scratch_dir = Path(cfg.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    bitrate = cfg.output_format.split("_")[-1] + "k"
    voice_settings = VoiceSettings(**dict(cfg.voice_settings))
    language_code = cfg.get("language_code")
    context_chars = int(cfg.get("context_chars", 300))
    seed = cfg.get("seed")

    client = ElevenLabs()
    succeeded, skipped, failed = 0, 0, 0
    for txt_path in txt_files:
        dst_path = dst_dir / txt_path.with_suffix(".mp3").name

        if dst_path.exists():
            print(f"{txt_path.name}: output already exists, skipping")
            skipped += 1
            continue

        try:
            text, marker_fixes = normalize_pause_markers(txt_path.read_text(encoding="utf-8"))
            if marker_fixes:
                print(f"{txt_path.name}: normalized {marker_fixes} non-canonical pause marker(s)")
            blocks = split_structural_blocks(text)
            tts_texts = [text_for_tts(spoken) for spoken, _ in blocks]

            with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
                work_dir = Path(tmp)
                pieces = []
                tts_calls = 0
                for i, (spoken, pause_seconds) in enumerate(blocks):
                    tts_text = tts_texts[i]
                    if tts_text:
                        prev = context_snippet(tts_texts[i - 1], context_chars, from_end=True) if i > 0 else None
                        nxt = context_snippet(tts_texts[i + 1], context_chars, from_end=False) if i + 1 < len(blocks) else None
                        audio = synthesize_segment(
                            client, tts_text, voice_id, cfg.model_id, cfg.output_format,
                            voice_settings, language_code,
                            previous_text=prev, next_text=nxt, seed=seed,
                        )
                        pieces.append(mp3_bytes_to_wav(audio, work_dir, f"seg_{i}"))
                        tts_calls += 1
                    if pause_seconds > 0:
                        pieces.append(silence_wav(pause_seconds, work_dir, f"pause_{i}"))

                concat_to_mp3(pieces, dst_path, work_dir, bitrate)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1
        print(f"{txt_path.name}: {tts_calls} TTS block(s), {dst_path.stat().st_size} bytes -> {dst_path}")

    print(f"\n{succeeded} synthesized, {skipped} skipped (already done), {failed} failed, out of {len(txt_files)} total")


if __name__ == "__main__":
    main()
