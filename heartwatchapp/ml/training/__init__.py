"""Activity-model training. Runs in its own environment
(heartwatchapp/ml/training/.venv, requirements-train.txt), never the app's.
dataset.py and baselines.py need only numpy; train.py and export.py need
TensorFlow. See the README's "Activity model" section."""
