"""Stub package - see ../pyproject.toml. Only satisfies faster-whisper's install-time dependency
declaration; any real use raises a clear error instead of silently misbehaving, in case an
assumption here ever changes (e.g. faster-whisper's vad_filter is ever set True somewhere)."""


def __getattr__(name: str):
    raise ImportError(
        "This is a stub 'onnxruntime' package, not the real library - no onnxruntime wheel is "
        "installable on this system (incompatible glibc baseline). It exists only to satisfy "
        "faster-whisper's install-time dependency declaration; faster-whisper only actually "
        "imports onnxruntime inside its own built-in VAD filter (vad_filter=True), which this "
        "project never enables (VAD is always done separately via torch.hub's Silero VAD). "
        f"Something just tried to access onnxruntime.{name} for real, which this stub can't "
        "provide - check why vad_filter ended up True somewhere."
    )
