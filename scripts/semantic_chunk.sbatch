#!/bin/bash
#SBATCH --job-name=concept_recognition_semantic_chunk            # Название задачи
#SBATCH --error=concept_recognition_semantic_chunk-%j.err        # Файл для вывода ошибок
#SBATCH --output=concept_recognition_semantic_chunk-%j.log       # Файл для вывода результатов
#SBATCH --time=24:00:00                     # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate concept_recognition

# Slurm starts the job in SLURM_SUBMIT_DIR (the directory `sbatch` was run from); make sure
# we're at the repo root regardless of where that was. Submit this job from the repo root, e.g.
# `sbatch scripts/semantic_chunk.sbatch` from inside ConceptRecognition/.
cd "$SLURM_SUBMIT_DIR"

# Model weights must already be present in models/sentence_transformers (run
# scripts/download_embedding_model.py once on a machine with internet access, then copy the
# whole repo, including models/, to the cluster).
if [ ! -d "models/sentence_transformers" ]; then
    echo "models/sentence_transformers not found. Run scripts/download_embedding_model.py with internet access first." >&2
    exit 1
fi

poetry run python scripts/semantic_chunk.py
