# Agent Development Guide

This guide explains how to build, test, and submit your Bomberland AI agent.

## 🤖 Baseline Agents
You can find several baseline agents in this directory to use as a starting point:
*   `random_agent.py`: Simple random actions.
*   `simple_rule_agent.py`: Avoids bombs and places bombs.
*   `tactical_rule_agent.py`: Uses BFS for pathfinding and targets enemies.
*   `dqn_agent/`: A Dueling Double-DQN implementation with a danger-aware state
    encoder, a safety/escape planner (hybrid rules+RL) and self-play training.
    Submission files: `agent.py` + `model.py` + `model.pt`. Training lives in
    `train.py` and is never imported by the submission.

## 🛠️ Developing Your Agent
Your agent must be a Python class named `Agent` inside a file named `agent.py`. It must implement an `act` method:

```python
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def act(self, obs: dict) -> int:
        # obs: dict containing 'map', 'players', 'bombs'
        # Returns: int in [0, 5]
        ...
```

### Constraints
*   **Time Limit**: 100ms per step.
*   **Resources**: CPU-only evaluation. No GPU access.
*   **Isolation**: No network access or file writing during the match.

## Training on Kaggle

The trainer is `agent/dqn_agent/train.py`. It must be run **as a module from the
repo root** (so `engine`, the baselines and the model share one clean import
path). Enable a GPU accelerator in the Kaggle notebook for the fastest learning.

1. GitHub token: github.com > Settings > Developer settings > Personal Access
   Tokens > Generate new token (read repo) > copy it.
2. Kaggle notebook > Settings > Secrets > add Key `dqn`, Value = the token.
3. Run these cells:
```python
# Cell 1 — token
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("dqn")

# Cell 2 — clone YOUR fork (where the trained code lives)
!git clone https://bphong-cyrus:{tok}@github.com/bphong-cyrus/Bomberland-GDGoC-AI-Challenge.git

# Cell 3 — deps + sanity
%cd /kaggle/working/Bomberland-GDGoC-AI-Challenge
!pip -q install trueskill
import torch; print("cuda:", torch.cuda.is_available())

# Cell 4 — train (self-play vs a mix of strong baselines + frozen self snapshots)
!python -m agent.dqn_agent.train \
    --episodes 20000 --opponents mix \
    --n_step 3 --batch_size 256 --buffer_cap 100000 \
    --self_play_after 2000 --snapshot_every 1000 --eval_every 1000 \
    --save_dir /kaggle/working/ckpts

# Cell 5 — download for submission
from IPython.display import FileLink
FileLink('ckpts/selfplay_mix_20000ep/model.pt')
```
Resume an interrupted run with `--resume ckpts/selfplay_mix_20000ep/epXXXX.pth`.

**Submission (flat zip):** put `agent.py`, `model.py` and the trained `model.pt`
in the zip root (no folder). The trained `model.pt` is TorchScript, so the
submission needs nothing but `numpy` + `torch`.

## 🧪 Local Testing

Guidance for local testing before submitting your agent. All participant scripts are located in the `scripts/participant/` folder.

### 1. Evaluate Agent Performance
To get a quick estimate of your agent's TrueSkill rating, run the ranking script. It will play matches against random strong baseline bots (Tactical, Smarter, Genius) and compute your estimated win rate and leaderboard score.
```bash
python -m scripts.participant.estimate_rankings --agent_path path/to/your/agent/ --num_matches 100
```

### 2. Run Headless or Visual Matches
Use the local match script to pit specific agents against each other or watch them play.
```bash
python -m scripts.participant.run_local_match --agent_paths path/to/your/agent/ None None None --visualize true
```

#### Arguments:
*   `--agent_paths`: Expects exactly 4 arguments representing the 4 players. You can pass:
    *   **A folder path** (e.g., `agent/dqn_agent/`): Perfect for Deep Reinforcement Learning agents. It will automatically load `agent.py` inside that folder, allowing your agent to load its weights relative to itself.
    *   **A file path** (e.g., `agent/random_agent.py`): Perfect for rule-based agents that don't need external weights.
    *   **A baseline name** (e.g., `TacticalRuleAgent`): Explicitly loads a built-in bot.
    *   `None` or `Random`: Automatically loads a random baseline bot.
*   `--visualize true`: Opens the PyGame window to watch the match live. Set to `false` for fast headless testing.

### 3. Estimate Agent Step Time
Ensure your agent doesn't violate the 100ms per step limit by running a timing benchmark:
```bash
python -m scripts.participant.estimate_agent_time path/to/your/agent/ --opponents None None None --num_matches 10
```

**Note:** This script doesn't ensure passing the 100ms per step limit on VM due to hardware difference.

### 4. Replay Saved Matches
If you downloaded a match log (`.json`) from the Google Drive, you can replay it:
```bash
python -m scripts.participant.replay_viewer path/to/log.json
```

## 📤 Submission Process
1.  **Package**: Create a `.zip` file containing:
    *   `agent.py` (Required)
    *   Any weights or models (e.g., `.pth` files)
2.  **Submit**: Use the Official Submission Form with your **Team ID** and **Token**.
3.  **Feedback**: Once submitted, your agent will immediately play 12 matches. You can check the leaderboard for your updated rating.
