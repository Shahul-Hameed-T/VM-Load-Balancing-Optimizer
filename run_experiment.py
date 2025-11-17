# src/run_experiment.py
import argparse, os, csv, pickle, json
from src.utils import ensure_dirs, seed_all, now_str
from src.workloads import steady, spike, bursty, session_locality
from src.simulator import run_sim

ensure_dirs()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", default="RR")
    parser.add_argument("--workload", default="steady")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim_time", type=float, default=600.0)
    parser.add_argument("--num_vms", type=int, default=10)
    parser.add_argument("--vm_cap", type=int, default=8192)
    parser.add_argument("--vm_cpu", type=float, default=4.0)
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--use_ml", action='store_true')
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--collect_for_ml", action='store_true')
    args = parser.parse_args()

    seed_all(args.seed)
    if args.workload == 'steady':
        arrivals = steady(args.sim_time, seed=args.seed)
    elif args.workload == 'spike':
        arrivals = spike(args.sim_time, seed=args.seed)
    elif args.workload == 'bursty':
        arrivals = bursty(args.sim_time, seed=args.seed)
    elif args.workload == 'session':
        arrivals = session_locality(args.sim_time, seed=args.seed)
    else:
        raise ValueError("unknown workload")

    out_prefix = f"{args.alg}_{args.workload}_s{args.seed}_{now_str()}"
    per_req, state, summary = run_sim(arrivals,
                                     num_vms=args.num_vms, vm_capacity=args.vm_cap,
                                     vm_capacity_cpu=args.vm_cpu,
                                     sim_time=args.sim_time, tick=args.tick,
                                     lb_name=args.alg, seed=args.seed,
                                     use_ml=args.use_ml, model_path=args.model_path)
    with open(f"data/{out_prefix}_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    with open(f"data/{out_prefix}_perreq.csv","w",newline='') as f:
        w = csv.writer(f)
        w.writerow(['status','arrival','req_ram','req_cpu','dur','client','vm'])
        w.writerows(per_req)
    with open(f"data/{out_prefix}_state.pkl","wb") as f:
        pickle.dump(state, f)
    print("Saved results with prefix:", out_prefix)

    if args.collect_for_ml:
        os.makedirs("data/ml_raw", exist_ok=True)
        with open(f"data/ml_raw/{out_prefix}.pkl","wb") as f:
            pickle.dump((per_req, state, summary), f)
        print("Appended run to data/ml_raw/")

if __name__ == '__main__':
    main()
