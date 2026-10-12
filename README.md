# Running the project

Clone the repo and follow below instructions
Use Python 3.11 in a separate virtual environment.
Run these commands from its folder.

Install dependencies:
python -m pip install -r requirements.txt

Run tests:
python -m pytest -q

Train all three seeds and evaluate:
python train.py
python evaluate.py
python verify_results.py
python plot_results.py

Create demonstrations:
python replay.py --workload nominal
python replay.py --workload robustness

Results are saved in outputs/.
Checkpoints and results from the submitted run are included.
To verify those saved results without retraining:
python verify_results.py