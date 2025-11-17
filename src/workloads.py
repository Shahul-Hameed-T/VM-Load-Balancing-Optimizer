# src/workloads.py
import numpy as np
import random

def steady(sim_time, rate=8.0, mean_ram=200, mean_cpu=0.5, mean_dur=5.0, seed=0):
    random.seed(seed); np.random.seed(seed)
    arrivals=[]
    t=0.0
    while t < sim_time:
        ia = np.random.exponential(1.0 / rate)
        t += ia
        req_ram = max(10, np.random.normal(mean_ram, mean_ram*0.4))
        req_cpu = max(0.01, np.random.normal(mean_cpu, mean_cpu*0.4))
        dur = max(0.1, np.random.exponential(mean_dur))
        arrivals.append((t, float(req_ram), float(req_cpu), float(dur), random.randint(0,1000000)))
    return arrivals

def spike(sim_time, base_rate=8.0, spike_start=200, spike_end=240, spike_factor=4.0,
          mean_ram=200, mean_cpu=0.5, mean_dur=5.0, seed=0):
    random.seed(seed); np.random.seed(seed)
    arrivals=[]
    t=0.0
    while t < sim_time:
        rate = base_rate
        if spike_start <= t <= spike_end:
            rate = base_rate * spike_factor
        ia = np.random.exponential(1.0 / rate)
        t += ia
        req_ram = max(10, np.random.normal(mean_ram, mean_ram*0.4))
        req_cpu = max(0.01, np.random.normal(mean_cpu, mean_cpu*0.4))
        dur = max(0.1, np.random.exponential(mean_dur))
        arrivals.append((t, float(req_ram), float(req_cpu), float(dur), random.randint(0,1000000)))
    return arrivals

def bursty(sim_time, base_rate=6.0, alpha=1.5, mean_ram=200, mean_cpu=0.5, mean_dur=5.0, seed=0):
    random.seed(seed); np.random.seed(seed)
    arrivals=[]
    t=0.0
    while t < sim_time:
        ia = (np.random.pareto(alpha) + 1) / base_rate
        t += ia
        req_ram = max(10, np.random.normal(mean_ram, mean_ram*0.6))
        req_cpu = max(0.01, np.random.normal(mean_cpu, mean_cpu*0.6))
        dur = max(0.1, np.random.exponential(mean_dur))
        arrivals.append((t, float(req_ram), float(req_cpu), float(dur), random.randint(0,1000000)))
    return arrivals

def session_locality(sim_time, n_clients=200, rate=8.0, p_repeat=0.7, mean_ram=200, mean_cpu=0.5, mean_dur=5.0, seed=0):
    random.seed(seed); np.random.seed(seed)
    arrivals=[]
    t=0.0
    while t < sim_time:
        ia = np.random.exponential(1.0 / rate)
        t += ia
        if random.random() < p_repeat:
            client = random.randint(0, n_clients//2)
        else:
            client = random.randint(0, n_clients)
        req_ram = max(10, np.random.normal(mean_ram, mean_ram*0.4))
        req_cpu = max(0.01, np.random.normal(mean_cpu, mean_cpu*0.4))
        dur = max(0.1, np.random.exponential(mean_dur))
        arrivals.append((t, float(req_ram), float(req_cpu), float(dur), client))
    return arrivals
