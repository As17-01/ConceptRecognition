import sys

import hydra

from pathlib import Path
from omegaconf import DictConfig
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings


@hydra.main(config_path="../conf", config_name="preview_voice", version_base=None)
def main(cfg: DictConfig) -> None:
    """Synthesizes one short snippet with the cloned voice so the voice + voice_settings can be
    auditioned before committing to a full (slow, billed) synthesize_speech.py run over every
    class. Keep voice_settings here in sync with synthesize_speech.yaml so the preview is
    representative. Re-run freely with overrides, e.g. voice_settings.speed=0.8 text="...".
    """
    voice_id_path = Path(cfg.voice_id_file)
    if not voice_id_path.is_file():
        print(f"Voice ID file not found: {voice_id_path} - run clone_voice.py first", file=sys.stderr)
        sys.exit(1)
    voice_id = voice_id_path.read_text(encoding="utf-8").strip()

    voice_settings = VoiceSettings(**dict(cfg.voice_settings))

    client = ElevenLabs()
    chunks = client.text_to_speech.convert(
        text=cfg.text,
        voice_id=voice_id,
        model_id=cfg.model_id,
        output_format=cfg.output_format,
        voice_settings=voice_settings,
        language_code=cfg.get("language_code"),
    )
    audio = b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))

    dst_path = Path(cfg.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(audio)
    print(f"Preview written -> {dst_path} ({len(audio)} bytes). Listen before running synthesize_speech.py.")


if __name__ == "__main__":
    main()
