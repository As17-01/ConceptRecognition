import subprocess
import sys

import hydra

from pathlib import Path
from omegaconf import DictConfig
from elevenlabs.client import ElevenLabs

# ElevenLabs Instant Voice Cloning accepts at most this many reference files per voice.
MAX_IVC_FILES = 25


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def select_source_files(src_dir: Path, sample_files, num_source_files: int) -> list[Path]:
    """Resolves which recordings to sample from. An explicit sample_files list wins; otherwise
    num_source_files recordings are picked evenly spread across the sorted recordings so the
    reference set spans different dates/sessions rather than clustering in one period."""
    if sample_files:
        paths = [src_dir / name for name in sample_files]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            print(f"Sample source file(s) not found: {', '.join(p.name for p in missing)}", file=sys.stderr)
            sys.exit(1)
        return paths

    all_files = sorted(src_dir.glob("*.mp3"))
    if not all_files:
        print(f"No .mp3 recordings found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    count = min(num_source_files, len(all_files))
    # Even spread across the corpus: index i*len/count lands on distinct, spaced-out recordings.
    return [all_files[i * len(all_files) // count] for i in range(count)]


def extract_clips(
    src_path: Path,
    dst_dir: Path,
    start_index: int,
    clips_per_file: int,
    clip_seconds: int,
    skip_start_seconds: float,
    early_window_seconds: float,
) -> tuple[list[Path], list[float]]:
    """Trims clips_per_file short reference clips from the opening of the recording, before
    background music usually starts. skip_start_seconds skips mic-check/greeting chatter at the
    very top; all clips land within early_window_seconds after that. start_index keeps clip
    filenames unique across multiple source recordings written into the same scratch dir."""
    duration = probe_duration(src_path)
    window_start = skip_start_seconds
    window_end = min(duration, window_start + early_window_seconds)
    if window_end - window_start < clip_seconds:
        print(
            f"  WARNING: {src_path.name} early window too short "
            f"({window_end - window_start:.0f}s) for {clip_seconds}s clips - skipping",
            file=sys.stderr,
        )
        return [], []

    # Spread clip start times across the early window (not the full hour-long file).
    latest_start = window_end - clip_seconds
    if clips_per_file == 1:
        offsets = [window_start]
    else:
        span = latest_start - window_start
        offsets = [window_start + span * i / (clips_per_file - 1) for i in range(clips_per_file)]

    dst_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, offset in enumerate(offsets):
        clip_path = dst_dir / f"sample_{start_index + i}.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(offset),
                "-i", str(src_path),
                "-t", str(clip_seconds),
                "-acodec", "libmp3lame", "-q:a", "2",
                str(clip_path),
            ],
            capture_output=True,
            check=True,
        )
        clips.append(clip_path)
    return clips, offsets


def resolve_selected_clips(scratch_dir: Path, selected_clips) -> list[Path]:
    """Loads a user-curated list of existing reference clips from scratch_dir. No extraction,
    no overwrites - just validates paths and returns them in listed order."""
    if not selected_clips:
        return []

    clips = []
    missing = []
    for name in selected_clips:
        path = scratch_dir / name
        if path.is_file():
            clips.append(path)
        else:
            missing.append(name)
    if missing:
        print(f"Selected clip(s) not found under {scratch_dir}: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    if len(clips) > MAX_IVC_FILES:
        print(
            f"Selected {len(clips)} clips, but ElevenLabs IVC allows at most {MAX_IVC_FILES}.",
            file=sys.stderr,
        )
        sys.exit(1)
    return clips


def upload_cloned_voice(client: ElevenLabs, clips: list[Path], cfg: DictConfig) -> None:
    # Pass (filename, bytes, content_type) tuples - bare path strings are treated as file *content*,
    # not paths, which makes ElevenLabs reject the upload as corrupted audio.
    files = [(clip.name, clip.read_bytes(), "audio/mpeg") for clip in clips]
    voice = client.voices.ivc.create(
        name=cfg.voice_name,
        description=cfg.voice_description,
        labels=dict(cfg.voice_labels),
        files=files,
    )

    dst_path = Path(cfg.dst_voice_id_file)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(voice.voice_id, encoding="utf-8")
    print(f"Cloned voice '{cfg.voice_name}' -> voice_id {voice.voice_id} (saved to {dst_path})")

@hydra.main(config_path="../conf", config_name="clone_voice", version_base=None)
def main(cfg: DictConfig) -> None:
    scratch_dir = Path(cfg.scratch_dir)
    selected = resolve_selected_clips(scratch_dir, cfg.get("selected_clips"))

    if selected:
        print(f"Using {len(selected)} cherry-picked clip(s) from {scratch_dir} (no re-extraction):")
        for clip in selected:
            print(f"  - {clip.name}")
        clips = selected
    else:
        src_dir = Path(cfg.src_dir)
        if not src_dir.is_dir():
            print(f"Source folder not found: {src_dir}", file=sys.stderr)
            sys.exit(1)

        sources = select_source_files(src_dir, cfg.sample_files, cfg.num_source_files)

        total_clips = len(sources) * cfg.clips_per_file
        if total_clips > MAX_IVC_FILES:
            print(
                f"Requested {total_clips} clips ({len(sources)} files x {cfg.clips_per_file}), "
                f"but ElevenLabs IVC allows at most {MAX_IVC_FILES}. Lower num_source_files or clips_per_file.",
                file=sys.stderr,
            )
            sys.exit(1)

        clips = []
        for src_path in sources:
            new_clips, offsets = extract_clips(
                src_path,
                scratch_dir,
                len(clips),
                cfg.clips_per_file,
                cfg.clip_seconds,
                cfg.skip_start_seconds,
                cfg.early_window_seconds,
            )
            clips.extend(new_clips)
            if new_clips:
                offset_str = ", ".join(f"{o:.0f}s" for o in offsets)
                print(f"Extracted {len(new_clips)} early-window clips from {src_path.name} (at {offset_str})")

        if not clips:
            print("No reference clips extracted - check skip_start_seconds / early_window_seconds", file=sys.stderr)
            sys.exit(1)

        print(f"Total: {len(clips)} reference clips from {len(sources)} recording(s)")

    client = ElevenLabs()
    upload_cloned_voice(client, clips, cfg)


if __name__ == "__main__":
    main()
