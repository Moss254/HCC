import os

folders = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "src/utils",
    "webapp/templates",
    "webapp/static/css",
    "webapp/static/js",
    "webapp/static/images",
    "models/trained_models",
    "models/preprocessing",
    "models/model_performance",
    "tests",
    "docs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✔ Project structure created successfully!")
