# Bayesian Nash Equilibrium Analysis for Multi-Robot Intersection Negotiation

Three robots reach an uncontrolled intersection at the same time. Each one is privately **Aggressive** or **Cautious**, and each has to choose **GO** or **YIELD** without talking to the others. If two or more go, they collide. This repo models that as a static Bayesian game, solves the symmetric equilibrium in closed form, and checks in simulation whether simple learners actually find it.

Authors: Mohanish Punj and Prakhar Dwivedi, SVNIT Surat (Department of Electronics and VLSI Engineering).

## The model in short

- Types are drawn independently with `P(Aggressive) = q` (the common prior). Each robot only knows its own type.
- A robot's payoff depends on its type, its action, and how many of the *other two* robots go:

| Action | Others going | Aggressive | Cautious |
|---|---|---|---|
| GO | 0 | +3 | +3 |
| GO | 1 | -5 | -8 |
| GO | 2 | -10 | -15 |
| YIELD | any | -1 | -0.5 |

- Each type has an indifference threshold on the chance `p` that a random opponent goes: `p*_A ≈ 0.2630` and `p*_C ≈ 0.1640`.
- The symmetric equilibrium follows from those thresholds. At `q = 0.5`, Cautious robots always yield and Aggressive robots go with probability **0.526**. That gives a **17.1%** collision probability, and a worse average payoff than letting one robot go at a time.

The payoff numbers and `q` are modelling assumptions. This is a simplified framework, not a validated model of real robot behaviour.

## What the code does

`bayesian_intersection_game.py` does four things:

1. Finds the two thresholds by bisection and computes the closed-form equilibrium for any `q`.
2. Runs a **model-based smoothed best-response learner**. Robots know the payoff table but learn how often others go. It starts both types at GO probability 0.9 (a deliberately unsafe start so there is a visible safety-improvement phase) and runs 15,000 rounds.
3. Runs **independent tabular Q-learning** (one Q-table per type, discount 0, epsilon-greedy with epsilon decaying from 0.20 to 0.03).
4. Saves four plots and a `results_summary.json` into `results/`.

## Running it

```bash
pip install -r requirements.txt
python bayesian_intersection_game.py
```

Everything is seeded (`SEED = 7`), so you should get the same numbers each time. Constants such as `Q_PRIOR`, `N_ROUNDS` and the step sizes sit at the top of the file.

## Results (q = 0.5, seed 7)

| Method | P(GO \| Aggressive) | P(GO \| Cautious) | Collision rate |
|---|---|---|---|
| Closed-form equilibrium | 0.526 | 0.000 | 17.1% |
| Smoothed best response (last 1000 rounds) | 0.548 | about 0 | 16.3% |
| Q-learning (last 1000 rounds, softmax estimate) | 0.267 | 0.201 | 1.4% |

The best-response learner settles around the equilibrium and keeps oscillating there, which is expected for a mixed equilibrium with constant step sizes. Q-learning does not reproduce it: it ends up yielding most of the time, because GO looked bad early and is then sampled too rarely to be corrected. The Q-learning curve is a softmax *diagnostic* for plotting, not the policy that is actually executed.

### Plots

| | |
|---|---|
| ![Learning](results/01_learning_convergence.png) | ![Collisions](results/02_collision_rate.png) |
| Model-based learning vs the equilibrium | Collision rate during learning |
| ![Equilibrium vs q](results/03_equilibrium_vs_q.png) | ![Q-learning](results/04_qlearning_convergence.png) |
| Equilibrium strategies as the prior `q` varies | Independent Q-learning |

## Limitations

- Payoffs and `q` are assumed, not measured.
- Convergence of the best-response learner is shown empirically here, not proven.
- Hyperparameters were not tuned, so the Q-learning result does not rule out a better-configured variant.
- Collisions are an abstract rule (two or more GO). There is no geometry, dynamics or sensing noise, and no physical robot experiment.
- Only the symmetric equilibrium is analysed.
