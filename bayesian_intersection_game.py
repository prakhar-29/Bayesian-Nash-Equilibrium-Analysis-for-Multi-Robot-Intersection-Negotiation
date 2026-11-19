from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

SEED = 7
Q_PRIOR = 0.5
N_ROUNDS = 15_000
ALPHA_STRATEGY = 0.01
ALPHA_BELIEF = 0.02
ROLLING_WINDOW = 200
RESULTS_DIR = Path("results")

# Payoffs indexed by number of OTHER robots that choose GO.
PAYOFFS = {
    "Aggressive": {"GO": [3.0, -5.0, -10.0], "YIELD": -1.0},
    "Cautious":   {"GO": [3.0, -8.0, -15.0], "YIELD": -0.5},
}


def go_payoff(kind, p):
    """Expected GO payoff when each opponent independently chooses GO with prob p."""
    if kind == "Aggressive":
        return 3.0 * (1-p)**2 + 2*p*(1-p)*(-5.0) + p**2*(-10.0)
    return 3.0 * (1-p)**2 + 2*p*(1-p)*(-8.0) + p**2*(-15.0)


def yield_payoff(kind):
    return PAYOFFS[kind]["YIELD"]


def threshold(kind):
    """Numerically find p where expected GO payoff equals YIELD payoff."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        diff = go_payoff(kind, mid) - yield_payoff(kind)
        if diff > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


P_STAR_A = threshold("Aggressive")
P_STAR_C = threshold("Cautious")


def best_response(kind, p, tol=1e-10):
    """Pure best response. At indifference, return None (any mix is a best response)."""
    diff = go_payoff(kind, p) - yield_payoff(kind)
    if abs(diff) <= tol:
        return None
    return 1.0 if diff > 0 else 0.0


def analytical_equilibrium(q):
    """Closed-form symmetric BNE for the specified payoffs."""
    if not (0 <= q <= 1):
        raise ValueError("q must be in [0, 1]")
    if q < P_STAR_C:
        g_a = 1.0
        g_c = (P_STAR_C - q) / (1.0 - q) if q < 1 else 0.0
    elif q < P_STAR_A:
        g_a, g_c = 1.0, 0.0
    else:
        g_a = P_STAR_A / q if q > 0 else 1.0
        g_c = 0.0
    p = q * g_a + (1-q) * g_c
    return {"q": q, "g_A": float(g_a), "g_C": float(g_c), "p": float(p)}


def rolling_mean(x, window):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window)/window, mode="valid")


def run_smoothed_best_response(q=Q_PRIOR, n_rounds=N_ROUNDS, seed=SEED):
    """
    Model-based learning, inspired by smoothed best response/fictitious play.
    The agents know the payoff model, but estimate aggregate GO frequency from outcomes.
    Initial GO probability and belief are set to 0.9 to model initially unsafe behavior.
    """
    rng = np.random.default_rng(seed)
    # Start with an unsafe/high-conflict policy so learning has a visible safety-improvement phase.
    g = {"Aggressive": 0.9, "Cautious": 0.9}
    p_hat = 0.9

    g_a_hist = np.zeros(n_rounds)
    g_c_hist = np.zeros(n_rounds)
    collision_hist = np.zeros(n_rounds)
    go_fraction_hist = np.zeros(n_rounds)

    for t in range(n_rounds):
        types = np.where(rng.random(3) < q, "Aggressive", "Cautious")
        probs = np.array([g[k] for k in types], dtype=float)
        actions_go = rng.random(3) < probs
        n_go = int(actions_go.sum())
        collision_hist[t] = float(n_go >= 2)
        go_fraction_hist[t] = n_go / 3.0

        # Recency-weighted estimate from observed aggregate actions.
        p_hat += ALPHA_BELIEF * (n_go / 3.0 - p_hat)
        p_hat = float(np.clip(p_hat, 0.0, 1.0))

        # Update the strategy for each type present this round.
        # This makes the update sample-based; types absent in a round are not updated.
        for kind in ("Aggressive", "Cautious"):
            if np.any(types == kind):
                br = best_response(kind, p_hat)
                # At exact indifference, preserve the current mixed strategy.
                if br is not None:
                    g[kind] += ALPHA_STRATEGY * (br - g[kind])
                    g[kind] = float(np.clip(g[kind], 0.0, 1.0))

        g_a_hist[t] = g["Aggressive"]
        g_c_hist[t] = g["Cautious"]

    return {
        "g_A": g_a_hist,
        "g_C": g_c_hist,
        "collision": collision_hist,
        "go_fraction": go_fraction_hist,
        "p_hat": p_hat,
    }


def run_q_learning(q=Q_PRIOR, n_rounds=N_ROUNDS, seed=SEED,
                   learning_rate=0.05, epsilon_start=0.20, epsilon_end=0.03):
    """
    Independent tabular Q-learning.
    State = private type; actions = GO(1) or YIELD(0).
    Since each round is treated as a one-shot game, gamma=0 and Q updates toward reward.
    Epsilon decays linearly but remains nonzero to keep exploring.
    """
    rng = np.random.default_rng(seed)
    # Q[kind, action], action index 0=YIELD, 1=GO
    q_values = {
        "Aggressive": np.array([0.0, 0.0]),
        "Cautious": np.array([0.0, 0.0]),
    }
    counts = {
        "Aggressive": np.zeros(2, dtype=int),
        "Cautious": np.zeros(2, dtype=int),
    }
    # Smoothed action-frequency estimates per type for plotting policy learning.
    action_rate = {"Aggressive": 0.5, "Cautious": 0.5}
    hist_a = np.zeros(n_rounds)
    hist_c = np.zeros(n_rounds)
    collision = np.zeros(n_rounds)
    reward_a = np.full(n_rounds, np.nan)
    reward_c = np.full(n_rounds, np.nan)

    for t in range(n_rounds):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (t / max(1, n_rounds-1))
        types = np.where(rng.random(3) < q, "Aggressive", "Cautious")
        actions = np.zeros(3, dtype=int)  # 0=YIELD, 1=GO

        for i, kind in enumerate(types):
            if rng.random() < epsilon:
                actions[i] = int(rng.integers(0, 2))
            else:
                # Random tie-break while Q values are equal.
                best = np.flatnonzero(np.isclose(q_values[kind], q_values[kind].max()))
                actions[i] = int(rng.choice(best))

        n_go = int(actions.sum())
        collision[t] = float(n_go >= 2)

        # Rewards based on number of OTHER robots going.
        for i, kind in enumerate(types):
            others_go = n_go - actions[i]
            if actions[i] == 1:
                reward = PAYOFFS[kind]["GO"][others_go]
            else:
                reward = PAYOFFS[kind]["YIELD"]
            action_idx = actions[i]
            counts[kind][action_idx] += 1
            # One-shot Q-learning update: gamma=0.
            q_values[kind][action_idx] += learning_rate * (reward - q_values[kind][action_idx])
            if kind == "Aggressive":
                reward_a[t] = reward
            else:
                reward_c[t] = reward

        # Plot a smoothed empirical policy: frequency of GO among the last observations,
        # separately by type. This is not a mixed-policy parameter like the analytical g.
        for kind in ("Aggressive", "Cautious"):
            if np.any(types == kind):
                # Softmax policy gives a probability even with learned Q-values.
                diff = q_values[kind][1] - q_values[kind][0]
                temperature = 0.5
                z = np.clip(diff / temperature, -30, 30)
                target = 1.0 / (1.0 + np.exp(-z))
                action_rate[kind] += 0.05 * (target - action_rate[kind])

        hist_a[t] = action_rate["Aggressive"]
        hist_c[t] = action_rate["Cautious"]

    return {
        "g_A": hist_a,
        "g_C": hist_c,
        "collision": collision,
        "Q": {k: v.tolist() for k, v in q_values.items()},
        "counts": {k: v.tolist() for k, v in counts.items()},
    }


def plot_results(model, q_learning, q=Q_PRIOR):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eq = analytical_equilibrium(q)
    rounds = np.arange(1, len(model["g_A"]) + 1)

    # Plot 1: model-based strategy learning.
    plt.figure(figsize=(10, 5.5))
    plt.plot(rounds, model["g_A"], label="Aggressive learned GO probability", linewidth=1)
    plt.plot(rounds, model["g_C"], label="Cautious learned GO probability", linewidth=1)
    plt.axhline(eq["g_A"], linestyle="--", label=f"Aggressive BNE = {eq['g_A']:.3f}")
    plt.axhline(eq["g_C"], linestyle="--", label=f"Cautious BNE = {eq['g_C']:.3f}")
    plt.xlabel("Round")
    plt.ylabel("Probability of GO")
    plt.title("Model-based learning vs Bayesian Nash Equilibrium")
    plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_learning_convergence.png", dpi=160)
    plt.close()

    # Plot 2: collision rate over time.
    window = ROLLING_WINDOW
    collision_smooth = rolling_mean(model["collision"], window)
    plt.figure(figsize=(10, 5.5))
    plt.plot(np.arange(window, len(model["collision"]) + 1), collision_smooth,
             label=f"Observed rolling collision rate ({window}-round window)", linewidth=1)
    initial_collision = 3 * 0.9**2 - 2 * 0.9**3
    plt.scatter([0], [initial_collision], marker="o", zorder=5,
                label=f"Initial expected rate (GO probability 0.9) = {initial_collision:.1%}")
    plt.plot([0, window], [initial_collision, collision_smooth[0]],
             linestyle=":", linewidth=1, label="Transition to first rolling estimate")
    theoretical_collision = 3 * eq["p"]**2 - 2 * eq["p"]**3
    plt.axhline(theoretical_collision, linestyle="--",
                label=f"Theoretical BNE = {theoretical_collision:.3f}")
    plt.xlabel("Round")
    plt.ylabel("Collision rate")
    plt.title("Collision rate during model-based learning")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_collision_rate.png", dpi=160)
    plt.close()

    # Plot 3: equilibrium strategies vs q.
    q_grid = np.linspace(0.01, 1.0, 300)
    eqs = [analytical_equilibrium(float(qi)) for qi in q_grid]
    plt.figure(figsize=(10, 5.5))
    plt.plot(q_grid, [e["g_A"] for e in eqs], label="Aggressive: P(GO | Aggressive)")
    plt.plot(q_grid, [e["g_C"] for e in eqs], label="Cautious: P(GO | Cautious)")
    plt.axvline(P_STAR_C, linestyle=":", label=f"Cautious threshold = {P_STAR_C:.3f}")
    plt.axvline(P_STAR_A, linestyle=":", label=f"Aggressive threshold = {P_STAR_A:.3f}")
    plt.xlabel("Prior probability q = P(Aggressive)")
    plt.ylabel("Equilibrium probability of GO")
    plt.title("Bayesian Nash Equilibrium strategies vs prior q")
    plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_equilibrium_vs_q.png", dpi=160)
    plt.close()

    # Plot 4: Q-learning policy estimate vs BNE reference.
    plt.figure(figsize=(10, 5.5))
    plt.plot(rounds, q_learning["g_A"], label="Q-learning Aggressive softmax policy estimate", linewidth=1)
    plt.plot(rounds, q_learning["g_C"], label="Q-learning Cautious softmax policy estimate", linewidth=1)
    plt.axhline(eq["g_A"], linestyle="--", label=f"Aggressive BNE = {eq['g_A']:.3f}")
    plt.axhline(eq["g_C"], linestyle="--", label=f"Cautious BNE = {eq['g_C']:.3f}")
    plt.xlabel("Round")
    plt.ylabel("Estimated probability of GO")
    plt.title("Independent Q-learning (illustrative; not guaranteed to converge to BNE)")
    plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_qlearning_convergence.png", dpi=160)
    plt.close()

    summary = {
        "seed": SEED,
        "q": q,
        "rounds": N_ROUNDS,
        "thresholds": {"aggressive": P_STAR_A, "cautious": P_STAR_C},
        "analytical_equilibrium": eq,
        "theoretical_collision_probability": theoretical_collision,
        "model_based_last_1000_mean": {
            "g_A": float(np.mean(model["g_A"][-1000:])),
            "g_C": float(np.mean(model["g_C"][-1000:])),
            "collision_rate": float(np.mean(model["collision"][-1000:])),
        },
        "q_learning_last_1000_mean": {
            "g_A_policy_estimate": float(np.mean(q_learning["g_A"][-1000:])),
            "g_C_policy_estimate": float(np.mean(q_learning["g_C"][-1000:])),
            "collision_rate": float(np.mean(q_learning["collision"][-1000:])),
        },
        "q_learning_Q_values": q_learning["Q"],
        "q_learning_action_counts": q_learning["counts"],
    }
    with open(RESULTS_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("Thresholds from payoff model:")
    print(f"  Aggressive p* = {P_STAR_A:.6f}")
    print(f"  Cautious   p* = {P_STAR_C:.6f}")
    eq = analytical_equilibrium(Q_PRIOR)
    print("\nAnalytical BNE at q=0.5:")
    print(json.dumps(eq, indent=2))
    print(f"Theoretical collision probability = {3*eq['p']**2 - 2*eq['p']**3:.4f}")

    model = run_smoothed_best_response()
    q_learning = run_q_learning()
    summary = plot_results(model, q_learning)
    print("\nSimulation summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nPlots and summary saved in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
