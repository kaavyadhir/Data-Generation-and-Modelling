import numpy as np
from scipy.integrate import odeint


def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def run_simulation(beta, gamma, population, initial_infected, days=100):
    S0 = population - initial_infected
    I0 = initial_infected
    R0 = 0

    y0 = [S0, I0, R0]
    t = np.linspace(0, days, days)

    solution = odeint(sir_model, y0, t, args=(beta, gamma, population))
    S, I, R = solution.T

    return t, S, I, R