# src/simulator.py
import random, math
import numpy as np
import joblib
from src.utils import ensure_dirs
ensure_dirs()

class VM:
    def __init__(self, id, capacity_mb=8192, capacity_cpu=4.0):
        self.id = id
        self.capacity = float(capacity_mb)       # RAM capacity in MB
        self.capacity_cpu = float(capacity_cpu)  # vCPU capacity (e.g., 4 CPUs)
        self.inflight = []  # list of dicts {'end_time','ram','cpu'}
        self.history = []   # list of (time, ram_usage, cpu_usage)

    def prune(self, now):
        self.inflight = [it for it in self.inflight if it['end_time'] > now]

    def current_ram(self, now):
        self.prune(now)
        return sum(it['ram'] for it in self.inflight)

    def current_cpu(self, now):
        self.prune(now)
        return sum(it['cpu'] for it in self.inflight)

    def add_request(self, now, req_ram, req_cpu, duration):
        self.inflight.append({'end_time': now + duration, 'ram': float(req_ram), 'cpu': float(req_cpu)})

# ---------------- Load balancers ----------------
def lb_rr(state, req, now):
    i = state['rr_ptr'] % len(state['vms'])
    state['rr_ptr'] += 1
    return state['vms'][i]

def lb_lc(state, req, now):
    return min(state['vms'], key=lambda v: len(v.inflight))

def lb_leastram(state, req, now):
    return min(state['vms'], key=lambda v: v.current_ram(now))

def lb_leastcpu(state, req, now):
    return min(state['vms'], key=lambda v: v.current_cpu(now))

def lb_least_resource_combined(state, req, now):
    vms = state['vms']
    scores = []
    for v in vms:
        ram_frac = v.current_ram(now) / max(1.0, v.capacity)
        cpu_frac = v.current_cpu(now) / max(1.0, v.capacity_cpu)
        scores.append(ram_frac + cpu_frac)
    min_idx = int(np.argmin(scores))
    return vms[min_idx]

def lb_po2(state, req, now):
    a,b = random.sample(state['vms'], 2)
    left = a.current_ram(now)/max(1.0,a.capacity) + a.current_cpu(now)/max(1.0,a.capacity_cpu)
    right = b.current_ram(now)/max(1.0,b.capacity) + b.current_cpu(now)/max(1.0,b.capacity_cpu)
    return a if left < right else b

def lb_ch(state, req, now):
    client_id = req[4]
    idx = int(client_id) % len(state['vms'])
    return state['vms'][idx]

LB_ALGOS = {
    'RR': lb_rr,
    'LC': lb_lc,
    'LRAM': lb_leastram,
    'LCPU': lb_leastcpu,
    'COMB': lb_least_resource_combined,
    'PO2': lb_po2,
    'CH': lb_ch
}

# ---------------- Simulation ----------------
def run_sim(arrivals, num_vms=10, vm_capacity=8192, vm_capacity_cpu=4.0,
            sim_time=600.0, tick=0.5, lb_name='RR', seed=0,
            use_ml=False, model_path=None, ctrl_int=1.0):
    random.seed(seed)
    np.random.seed(seed)
    state = {'vms':[VM(i, vm_capacity, vm_capacity_cpu) for i in range(num_vms)], 'rr_ptr':0}
    now = 0.0
    arrivals_idx = 0
    per_req_logs = []
    last_pred_time = -9999.0
    model = None
    if use_ml and model_path:
        model = joblib.load(model_path)

    def predict_future_resources(now_predict):
        if not model:
            return [(v.current_ram(now_predict), v.current_cpu(now_predict)) for v in state['vms']]
        X=[]
        for v in state['vms']:
            cr = v.current_ram(now_predict)
            cc = v.current_cpu(now_predict)
            cnt = len(v.inflight)
            X.append([cr, cc, cnt, v.capacity, v.capacity_cpu])
        import numpy as _np
        preds = model.predict(_np.array(X))  # shape (n_vms, 2)
        return [tuple(map(float, p)) for p in preds]

    while now < sim_time:
        while arrivals_idx < len(arrivals) and arrivals[arrivals_idx][0] <= now:
            t_arr, req_ram, req_cpu, dur, client = arrivals[arrivals_idx]
            arrivals_idx += 1

            preds = None
            if use_ml and model and (now - last_pred_time >= ctrl_int - 1e-9):
                preds = predict_future_resources(now + ctrl_int)
                last_pred_time = now

            accepted = False
            chosen_vm = None

            if use_ml and preds is not None:
                scores = []
                for i, v in enumerate(state['vms']):
                    pred_ram, pred_cpu = preds[i]
                    ram_after = pred_ram + req_ram
                    cpu_after = pred_cpu + req_cpu
                    ram_ok = ram_after <= v.capacity
                    cpu_ok = cpu_after <= v.capacity_cpu
                    score = (ram_after / v.capacity) + (cpu_after / v.capacity_cpu)
                    if not (ram_ok and cpu_ok):
                        score += 1000.0
                    scores.append((score, i))
                scores.sort()
                best = scores[0]
                if best[0] < 900.0:
                    chosen_vm = state['vms'][best[1]]; accepted=True
                else:
                    per_req_logs.append(('rej', float(t_arr), float(req_ram), float(req_cpu), float(dur), int(client), None))
                    continue
            else:
                vm = LB_ALGOS.get(lb_name, lb_rr)(state, (t_arr, req_ram, req_cpu, dur, client), now)
                if vm.current_ram(now) + req_ram <= vm.capacity and vm.current_cpu(now) + req_cpu <= vm.capacity_cpu:
                    chosen_vm = vm; accepted=True
                else:
                    per_req_logs.append(('rej', float(t_arr), float(req_ram), float(req_cpu), float(dur), int(client), vm.id))
                    continue

            chosen_vm.add_request(now, req_ram, req_cpu, dur)
            per_req_logs.append(('ok', float(t_arr), float(req_ram), float(req_cpu), float(dur), int(client), chosen_vm.id))

        for v in state['vms']:
            state_ram = v.current_ram(now)
            state_cpu = v.current_cpu(now)
            v.history.append((now, float(state_ram), float(state_cpu)))

        next_arrival = arrivals[arrivals_idx][0] if arrivals_idx < len(arrivals) else sim_time + 1
        next_finish = min([it['end_time'] for v in state['vms'] for it in v.inflight], default=sim_time+1)
        next_time = min(now + tick, next_arrival, next_finish, sim_time)
        now = next_time

    total = len(per_req_logs)
    rejected = sum(1 for r in per_req_logs if r[0]=='rej')
    avg_ram = float(np.mean([np.mean([h for _,h,_ in v.history]) if len(v.history)>0 else 0.0 for v in state['vms']]))
    avg_cpu = float(np.mean([np.mean([c for _,_,c in v.history]) if len(v.history)>0 else 0.0 for v in state['vms']]))
    std_ram = float(np.std([np.mean([h for _,h,_ in v.history]) if len(v.history)>0 else 0.0 for v in state['vms']]))
    std_cpu = float(np.std([np.mean([c for _,_,c in v.history]) if len(v.history)>0 else 0.0 for v in state['vms']]))
    overload_ram_events = sum(1 for v in state['vms'] for t,r,c in v.history if r > 0.9 * v.capacity)
    overload_cpu_events = sum(1 for v in state['vms'] for t,r,c in v.history if c > 0.9 * v.capacity_cpu)
    summary = {'total_requests': total, 'rejected': rejected,
               'avg_ram': avg_ram, 'std_ram': std_ram,
               'avg_cpu': avg_cpu, 'std_cpu': std_cpu,
               'overload_ram_events': int(overload_ram_events),
               'overload_cpu_events': int(overload_cpu_events)}
    return per_req_logs, state, summary
    