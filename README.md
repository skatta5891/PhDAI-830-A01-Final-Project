# Running the project

Use Python 3.11 in a separate virtual environment. Clone the repository and run these commands from the project folder.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run tests:
```bash
python -m pytest -q
```

Train and evaluate:
```bash
python train.py
python evaluate.py
python verify_results.py
python plot_results.py
```

Create demonstrations:
```bash
python replay.py --workload nominal
python replay.py --workload robustness
```

Results are saved in `outputs/`. Submitted checkpoints and results are included.

Verify saved results without retraining:
```bash
python verify_results.py
```