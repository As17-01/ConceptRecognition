import sys

import hydra

from pathlib import Path
from omegaconf import DictConfig
from elevenlabs.client import ElevenLabs


def synthesize(client: ElevenLabs, text: str, voice_id: str, model_id: str, output_format: str) -> bytes:
    # convert() returns Iterator[bytes] (a streaming response) - not a single bytes object -
    # so the audio has to be assembled from chunks rather than written directly.
    chunks = client.text_to_speech.convert(text=text, voice_id=voice_id, model_id=model_id, output_format=output_format)
    return b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))


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

    client = ElevenLabs()
    succeeded, skipped, failed = 0, 0, 0
    for txt_path in txt_files:
        dst_path = dst_dir / txt_path.with_suffix(".mp3").name

        # Same resumability pattern as the rest of the pipeline: a crash or interruption
        # partway through a batch shouldn't force re-synthesizing (and re-billing) classes
        # that already finished.
        if dst_path.exists():
            print(f"{txt_path.name}: output already exists, skipping")
            skipped += 1
            continue

        try:
            text = txt_path.read_text(encoding="utf-8")
            audio = synthesize(client, text, voice_id, cfg.model_id, cfg.output_format)
            dst_path.write_bytes(audio)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        succeeded += 1
        print(f"{txt_path.name}: {len(audio)} bytes -> {dst_path}")

    print(f"\n{succeeded} synthesized, {skipped} skipped (already done), {failed} failed, out of {len(txt_files)} total")


if __name__ == "__main__":
    main()
