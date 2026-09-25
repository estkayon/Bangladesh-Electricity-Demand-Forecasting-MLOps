import os
from pathlib import Path

list_of_files = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "notebooks/.gitkeep",

    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/collect_data.py",
    "src/data/preprocess.py",

    "src/features/__init__.py",
    "src/features/build_features.py",

    "src/models/__init__.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/predict.py",

    "src/api/__init__.py",
    "src/api/app.py",

    "tests/__init__.py",

    "config/.gitkeep",

    "artifacts/.gitkeep",

    ".github/workflows/.gitkeep",

    ".gitignore",
    "requirements.txt",
    "README.md",
    "Dockerfile",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists():
        filepath.touch()
        print(f"Created: {filepath}")