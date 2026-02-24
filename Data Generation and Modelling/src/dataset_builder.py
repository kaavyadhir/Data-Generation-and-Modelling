import numpy as np
import pandas as pd
from simulation import run_simulation


def generate_dataset(num_samples=1000):
    data = []

    for _ in range(num_samples):
        beta = np.random.uniform(0.1, 0.6)
        gamma = np.random.uniform(0.05, 0.3)
        population = 1000
        initial_infected = np.random.randint(1, 50)

        t, S, I, R = run_simulation(beta, gamma, population, initial_infected)

        peak_infected = np.max(I)
        time_to_peak = t[np.argmax(I)]
        total_infected = R[-1]

        # Classification label
        severity = 1 if peak_infected > 400 else 0

        data.append([
            beta,
            gamma,
            initial_infected,
            peak_infected,
            time_to_peak,
            total_infected,
            severity
        ])

    columns = [
        "beta",
        "gamma",
        "initial_infected",
        "peak_infected",
        "time_to_peak",
        "total_infected",
        "severity"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv("../data/sir_dataset.csv", index=False)

    return df