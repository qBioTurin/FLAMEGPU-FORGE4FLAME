import json
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# SEIRS model ODEs
def seirs_model(y, t, beta, gamma, sigma, xi, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N + xi * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - xi * R
    return dSdt, dEdt, dIdt, dRdt

def generate_seirs_curve(days=105):
    beta = np.random.uniform(0.2, 0.9)    # transmission
    gamma = 0.2                            # recovery
    sigma = 0.2                            # incubation fixed
    xi = 0.001826484                       # S to R loss rate fixed
    N = 1e6
    S0 = N - 1
    E0 = 0
    I0 = 1
    R0 = 0
    y0 = (S0, E0, I0, R0)
    t = np.linspace(0, days-1, days)
    ret = odeint(seirs_model, y0, t, args=(beta, gamma, sigma, xi, N))
    S, E, I, R = ret.T
    fraction_infected = I / N
    data = [{"day": int(day+1), "percentage_infected": f"{val:.12f}"} for day, val in enumerate(fraction_infected)]
    return data, beta, gamma, sigma, xi

def generate_files(input_path, output_dir="NODE", n_files=500):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r") as f:
        base_data = json.load(f)

    for i in range(1, n_files + 1):
        new_curve, beta, gamma, sigma, xi = generate_seirs_curve()
        base_data["outside_contagion"] = new_curve
        output_path = f"{output_dir}/outside_contagion_{i}/model.json"
        os.makedirs(f"{output_dir}/outside_contagion_{i}", exist_ok=True)
        with open(output_path, "w") as out_f:
            json.dump(base_data, out_f, indent = 2)

        days = [d["day"] for d in new_curve]
        values = [float(d["percentage_infected"]) for d in new_curve]

        plt.figure(figsize=(10, 5))
        plt.plot(days, values, label="fraction_infected (SEIRS)")
        plt.title(f"Outside Contagion SEIRS {i}\nβ={beta:.3f} γ={gamma:.3f} σ={sigma:.3f} ξ={xi:.6f}")
        plt.xlabel("Day")
        plt.ylabel("Fraction Infected")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fname = f"{output_dir}/outside_contagion_{i}.png"
        # plt.savefig(fname)
        plt.close()

# Esempio d'uso:
generate_files("resources/f4f/Hospital_NoCountermeasures/model.json")