#!/bin/bash
#SBATCH --job-name=concept_recognition_transcribe                # Название задачи
#SBATCH --error=concept_recognition_transcribe-%j.err            # Файл для вывода ошибок
#SBATCH --output=concept_recognition_transcribe-%j.log           # Файл для вывода результатов
#SBATCH --time=24:00:00                     # Максимальное время выполнения
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate <env_name>

poetry run python scripts/transcribe.py --config-name=transcribe_obrazovatelny_kanal
