#!/bin/bash
#SBATCH --job-name=concept_recognition_transcribe                # Название задачи
#SBATCH --error=concept_recognition_transcribe-%j.err            # Файл для вывода ошибок
#SBATCH --output=concept_recognition_transcribe-%j.log           # Файл для вывода результатов
#SBATCH --partition=rocky
#SBATCH --time=23:00:00                     # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate concept_recognition

# Slurm starts the job in SLURM_SUBMIT_DIR (the directory `sbatch` was run from); make sure
# we're at the repo root regardless of where that was. Submit this job from the repo root, e.g.
# `sbatch scripts/transcribe/transcribe.sbatch` from inside ConceptRecognition/.
cd "$SLURM_SUBMIT_DIR"

# Model weights must already be present in models/whisper and models/silero_vad (run
# scripts/transcribe/download_whisper_model.py and scripts/transcribe/download_vad_model.py once
# on a machine with internet access, then copy the whole repo, including models/, to the cluster).
# faster-whisper caches the converted model via huggingface_hub's standard layout
# (models--<org>--<repo>/snapshots/<commit-hash>/model.bin) rather than a single flat file, and
# the commit hash isn't known ahead of time, so this checks for any model.bin under that layout.
if ! compgen -G "models/whisper/models--*/snapshots/*/model.bin" > /dev/null; then
    echo "No cached model.bin found under models/whisper. Run scripts/transcribe/download_whisper_model.py with internet access first." >&2
    exit 1
fi
if [ ! -d "models/silero_vad" ]; then
    echo "models/silero_vad not found. Run scripts/transcribe/download_vad_model.py with internet access first." >&2
    exit 1
fi

# ctranslate2's bundled cuDNN only ships the top-level dispatcher library; cuDNN 9.x's actual
# operation libraries (libcudnn_ops.so.9 etc.) are dlopen'd at runtime and won't be found unless
# their directory is on LD_LIBRARY_PATH (see pyproject.toml's nvidia-cudnn-cu12 comment for the
# full explanation). Resolved dynamically since the pip install location varies by environment.
CUDNN_LIB_DIR=$(poetry run python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))")
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$LD_LIBRARY_PATH"

poetry run python scripts/transcribe/transcribe.py
