"""Activity-model training. Runs in its own environment
(heartwatchapp/ml/training/.venv, requirements-train.txt), never the app's.
dataset.py, baselines.py and folds.py need only numpy; train.py,
train_v1.py and export.py import TensorFlow only inside the functions that
train or convert (so --folds-only and the tests need no TensorFlow). See
the README's "Activity model" section."""
