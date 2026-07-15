#!/bin/bash
#SBATCH --job-name=concept_recognition_preprocess                # Название задачи
#SBATCH --error=concept_recognition_preprocess-%j.err            # Файл для вывода ошибок
#SBATCH --output=concept_recognition_preprocess-%j.log           # Файл для вывода результатов
#SBATCH --partition=rocky
#SBATCH --time=24:00:00                     # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов

module load Python
source activate concept_recognition

# Slurm starts the job in SLURM_SUBMIT_DIR (the directory `sbatch` was run from); make sure
# we're at the repo root regardless of where that was. Submit this job from the repo root, e.g.
# `sbatch scripts/preprocess_transcripts/preprocess_transcripts.sbatch` from inside ConceptRecognition/.
cd "$SLURM_SUBMIT_DIR"

poetry run python scripts/preprocess_transcripts/preprocess_transcripts.py
