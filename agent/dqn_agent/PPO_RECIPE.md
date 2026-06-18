# PPO training recipe — 2026-06-18 (top-8 push)

Honest framing first, because the project history is unambiguous:

- The PPO+BC+shield lineage **plateaued at ~mu 114** and the proven ceiling is `it6000`
  (Variant-C, train-vs-champions, **no aggression**) which reached rank 8.
- **Aggression reward tuning failed 3×** on the real board (overlay, farm→hunt v1, v2):
  every one turned the policy reckless and it died ~step 365–382 vs the proven ~447.
  Survival dominates TrueSkill mu, so trading it away loses.
- So the **only evidence-backed +EV training move** is a *fresh-seed* reproduction of
  the proven **no-aggression** regime, plus the one causally-validated lever
  (**train-vs-champions**, `--opponent_ckpts`) and the new **`--tough_pool`** opponent.
- The search agent (`submit_search` / `submit_surv`) is the **higher-EV** bet this round.
  Treat the PPO run as a parallel lottery ticket (~30–40% it beats it6000, +0.2–0.5 mu),
  not the main hope.

Run on Kaggle GPU. **Never `--resume`** (it skips BC + restores low lr/entropy and
regressed 4×). `--iters` is the TOTAL target, not a delta. Train FRESH, vary only the seed.

> **Step 0 — push first.** `--tough_pool` (the new search-bot opponent) lives in
> uncommitted code, so commit + push to `main` BEFORE you re-clone on Kaggle. If you
> don't want to push, just DROP `--tough_pool` from the command (it then runs on the
> current `main` with the rule+self pool — still valid, just without the novel diversity).
> SMOKE-VERIFIED locally (2026-06-18): both Recipe A and B run BC→PPO→eval→save end-to-end.

---

## Recipe A — PROVEN regime, fresh seed + tougher pool  (RECOMMENDED)

Reproduces the it6000 regime (no hunt reward) and adds the search bot as a novel
opponent so the net learns to survive+fight a *cornering* opponent — diversity the
rule-only pool never had.

```bash
python -m agent.dqn_agent.train_ppo \
  --iters 1500 --seed 2026 \
  --bc_pretrain 1 --bc_games 400 --bc_epochs 5 \
  --lr 2.5e-4 --ent_start 0.03 --ent_end 0.005 \
  --lam_e 0.1 --lam_adv 0.05 \
  --self_play_after 150 --self_opp_prob 0.5 --pool_cap 12 \
  --shield_horizon 6 \
  --bomb_enemy 0.10 --idle_w 0 --hunt_boost 1.0 \   # << the PROVEN no-aggression reward
  --tough_pool 1 \
  --eval_every 250 --save_dir ckpts_ppo_A
```

If you have your champion `.pth` on Kaggle (e.g. `it1000.pth` / the it6000 checkpoint),
add the validated +1mu lever:

```bash
  --opponent_ckpts it1000.pth        # (append to the command above)
```

Run a second seed in parallel as a pure variance bet (this is where the upside is):
`--seed 777 --save_dir ckpts_ppo_A2`.

---

## Recipe B — controlled farm→hunt nudge  (what you asked for; HIGHER RISK)

Only if you want to retest aggression. Keep it MILD and isolated to the farmed-out
phase (the unconditional/strong versions are exactly what died early). Gate hard.

```bash
python -m agent.dqn_agent.train_ppo \
  --iters 1500 --seed 2026 \
  --bc_pretrain 1 --bc_games 400 --bc_epochs 5 --bc_sources genius,genius,genius,hunter \
  --lr 2.5e-4 --ent_start 0.03 --ent_end 0.008 \
  --lam_e 0.1 --lam_adv 0.05 \
  --self_play_after 150 --shield_horizon 6 --tough_pool 1 \
  --bomb_enemy 0.15 --idle_w 0.02 --hunt_boost 1.5 \  # << mild hunt; do NOT go higher
  --eval_every 250 --save_dir ckpts_ppo_B
```

Knob meanings (all read by `event_reward`):
- `--bomb_enemy` reward for placing a bomb whose blast reaches an enemy (OUTCOME, not
  proximity — proximity pull is what got it killed). Proven 0.10; mild 0.15. **Don't exceed ~0.2.**
- `--hunt_boost` multiplier on `bomb_enemy` once `boxes_left <= 5`. 1.0 = off (proven);
  1.5 = mild. The 2.0–3.0 sweeps regressed.
- `--idle_w` farmed-out idle penalty. 0 = proven; 0.02 = mild anti-idle.
- `--lam_e` enemy-proximity shaping, power-scaled. **Keep 0.1** (raising it = the v1/v2 death).

---

## Gate BEFORE submitting (both recipes)

1. **Entropy kill-gate** (catches the over-peaked/brittle collapse that sank every
   fine-tune). Reject if mean < 0.30:
   ```bash
   python -m agent.dqn_agent.entropy_probe --model ckpts_ppo_A/ppo_selfplay_1500it/model.pt --games 15
   ```
2. **Behaviour vs MIX** — must match it6000 survival (~447 steps); reject if it dies earlier:
   ```bash
   python -m agent.dqn_agent.test_behavior_ppo --model ckpts_ppo_A/.../model.pt --opponents mix --games 30
   ```
   Want: survive ≳ it6000, died@ LATE, and ≥ it6000 on wins/kills. If it dies earlier
   than it6000 → it's the aggression-regression again → discard.
3. **Package** with the ready template `submit_ppo/` (VERIFIED 2026-06-18: loads via the
   grader file-path precheck + acts at 2.14 ms/step; uses the PLAIN shield, NOT the failed
   tactical overlay):
   ```bash
   cp ckpts_ppo_A/ppo_selfplay_1500it/model.pt submit_ppo/model.pt   # replace placeholder
   python -m agent.dqn_agent.verify_submit submit_ppo                # must print [OK]
   python - <<'PY'                                                   # zip FLAT (4 files)
   import zipfile, os
   with zipfile.ZipFile("submit_ppo.zip","w",zipfile.ZIP_DEFLATED) as z:
       for f in ("agent.py","model.py","ppo.py","model.pt"):
           z.write(os.path.join("submit_ppo", f), f)
   print(zipfile.ZipFile("submit_ppo.zip").namelist())
   PY
   ```
   Submit `submit_ppo.zip` as a **parallel** entry. Keep the search-agent entries active too.

## Kaggle flow
Re-clone `main` (so you get `--tough_pool` + the search bot), upload any `.pth` for
`--opponent_ckpts` as a dataset, run A (and A2). `/kaggle/working` is wiped on close, so
**download `model.pt` (and the `.pth`)** before ending the session, then package locally
with the `submit_ppo/` template above.
