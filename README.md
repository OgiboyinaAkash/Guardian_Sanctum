# Guardian's Sanctum

Multi-objective gridworld demonstrating Q-Learning, SARSA, and First-Visit Monte Carlo.

This folder contains two ways to run the Guardian's Sanctum:

- Server / Python mode (research-ready): uses `guardian_env.py` and the tabular agents in `agents/`.
- Client / Browser mode (interactive): open `index.html` to run the environment and lightweight JS agents entirely in the browser.

Files of interest:
- `guardian_env.py`: Gym-like environment implementing the Sanctum (sentinels, keys, door, treasure, fast-move noise).
- `agents/`: Tabular implementations for `q_learning.py`, `sarsa.py`, and `monte_carlo.py` (Python).
- `train.py`: Example Python runner for one algorithm.
- `compare.py`: Example Python script to compare the three algorithms.
- `index.html`: Interactive browser UI — environment editor, patrol editor, trainable JS agents and visualization.
- `requirements.txt`: Minimal Python deps (`numpy`, `matplotlib`) if you run Python scripts.

Quick start — Python (PowerShell):

```powershell
cd "d:\AKASH.O\Downloads\FAI Project\guardian_sanctum"
python -m pip install -r requirements.txt
python train.py --algo qlearning --episodes 1000
python compare.py
```

Quick start — Browser (recommended for fast experiments):

1. Open the UI directly (may work in some browsers):

	- Double-click `index.html` or open it with your browser.

2. Preferred: serve the folder with a simple HTTP server and open the page (recommended to avoid local-file restrictions):

```powershell
cd "d:\AKASH.O\Downloads\FAI Project\guardian_sanctum"
# Python 3
python -m http.server 8000
# Then open http://localhost:8000/index.html in your browser
```

What I added in the browser UI (`index.html`):
- Map editor: toggle walls, and place Start, Key, Door, Treasure tiles via the editor toolbar.
- Patrol editor: create and edit sentinel patrol routes by adding/removing patrol points on the grid; multiple patrols supported.
- Full hyperparameter controls: episodes, learning rate (alpha), discount factor (gamma), epsilon (exploration) are configurable and used by the JS agents.
- JS implementations of tabular Q-Learning, SARSA, and First-Visit Monte Carlo with simple training and plotting tools.

Next steps you might want:
- Save/load maps and patrols as JSON files for reproducible experiments.
- Export training logs and Q-tables as CSV/JSON for offline analysis.
- Add decay schedules for epsilon and alpha.
- Replace JS agents with server-side Python training and stream results to the UI.

If you want, I can update this README with examples of experiment setups and typical hyperparameters used in my tests.