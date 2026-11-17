"""Shop-floor scheduling with NSGA-II.

Assign jobs to machines of the matching type (milling / molding) and minimize two
objectives at once:
  1. makespan: total processing hours on the busiest machine
  2. energy (kWh): processing energy + idle energy of used machines until the
     last job finishes
Faster machines draw more power, and slow machines stretch the makespan and make
the others sit idle, so the objectives conflict and we get a Pareto front.
A plan is then picked by preference: knee point, fastest, or lowest energy.

Same method as in my undergraduate thesis (fast non-dominated sorting, crowding
distance, binary tournament, crossover, mutation, elitism), applied to a new problem.

Usage: python optimizer.py  (runs a demo and saves pareto.png)
"""
import json
import random
from pathlib import Path

BASE = Path(__file__).parent
SHOP_FILE = BASE / "data" / "shop.json"


def load_shop():
    return json.loads(SHOP_FILE.read_text(encoding="utf-8"))


def evaluate(plan, tasks, machines):
    """plan[i] is the machine index for job i. Returns (makespan, energy)."""
    load = [0.0] * len(machines)
    energy = 0.0
    for task, m in zip(tasks, plan):
        hours = task["hours"] / machines[m]["speed"]
        load[m] += hours
        energy += hours * machines[m]["power_kw"]
    makespan = max(load)
    for m, busy in enumerate(load):
        if busy > 0:  # a used machine stays on until the last job is done
            energy += (makespan - busy) * machines[m]["idle_kw"]
    return makespan, energy


def dominates(a, b):
    """Minimization: a is no worse in every objective and better in at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_sort(objs):
    """Return fronts as lists of indices; fronts[0] is the Pareto front."""
    n = len(objs)
    dominated_by_me = [[] for _ in range(n)]
    num_dominate_me = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if dominates(objs[p], objs[q]):
                dominated_by_me[p].append(q)
            elif dominates(objs[q], objs[p]):
                num_dominate_me[p] += 1
        if num_dominate_me[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated_by_me[p]:
                num_dominate_me[q] -= 1
                if num_dominate_me[q] == 0:
                    nxt.append(q)
        fronts.append(nxt)
        i += 1
    return fronts[:-1]


def crowding_distance(front, objs):
    """Larger distance = less crowded; keeps the front spread out."""
    dist = {i: 0.0 for i in front}
    for k in range(len(objs[front[0]])):
        ordered = sorted(front, key=lambda i: objs[i][k])
        lo, hi = objs[ordered[0]][k], objs[ordered[-1]][k]
        dist[ordered[0]] = dist[ordered[-1]] = float("inf")  # always keep the extremes
        if hi == lo:
            continue
        for j in range(1, len(ordered) - 1):
            dist[ordered[j]] += (objs[ordered[j + 1]][k] - objs[ordered[j - 1]][k]) / (hi - lo)
    return dist


def nsga2(tasks, machines, pop_size=80, generations=200, mutation_rate=0.1, seed=0):
    rng = random.Random(seed)
    # machines each job can run on
    choices = []
    for t in tasks:
        ok = [m for m, mach in enumerate(machines) if mach["type"] == t["type"]]
        if not ok:
            raise ValueError(f"no available machine for job {t['id']} ({t['type']})")
        choices.append(ok)

    def random_plan():
        return [rng.choice(c) for c in choices]

    def crossover(a, b):  # uniform crossover
        return [x if rng.random() < 0.5 else y for x, y in zip(a, b)]

    def mutate(plan):     # move some jobs to another machine of the same type
        return [rng.choice(c) if rng.random() < mutation_rate else m for m, c in zip(plan, choices)]

    pop = [random_plan() for _ in range(pop_size)]
    for _ in range(generations):
        objs = [evaluate(p, tasks, machines) for p in pop]
        rank, crowd = {}, {}
        for r, front in enumerate(non_dominated_sort(objs)):
            crowd.update(crowding_distance(front, objs))
            for i in front:
                rank[i] = r

        def better(i, j):  # binary tournament: lower rank wins, then larger crowding distance
            if rank[i] != rank[j]:
                return i if rank[i] < rank[j] else j
            return i if crowd[i] >= crowd[j] else j

        children = []
        while len(children) < pop_size:
            p1 = better(rng.randrange(pop_size), rng.randrange(pop_size))
            p2 = better(rng.randrange(pop_size), rng.randrange(pop_size))
            children.append(mutate(crossover(pop[p1], pop[p2])))

        # elitism: merge parents and children, refill by front, break ties by crowding
        merged = pop + children
        objs = [evaluate(p, tasks, machines) for p in merged]
        pop = []
        for front in non_dominated_sort(objs):
            if len(pop) + len(front) <= pop_size:
                pop += [merged[i] for i in front]
            else:
                dist = crowding_distance(front, objs)
                rest = sorted(front, key=lambda i: dist[i], reverse=True)
                pop += [merged[i] for i in rest[:pop_size - len(pop)]]
                break

    # final Pareto front, without duplicate objective values
    objs = [evaluate(p, tasks, machines) for p in pop]
    front, seen = [], set()
    for i in sorted(non_dominated_sort(objs)[0], key=lambda i: objs[i]):
        key = (round(objs[i][0], 2), round(objs[i][1], 1))
        if key not in seen:
            seen.add(key)
            front.append((pop[i], objs[i]))
    return front


def pick(front, priority="balanced"):
    """time: shortest makespan; energy: lowest energy; balanced: knee point
    (closest to the ideal point after normalizing both objectives)."""
    if priority == "time":
        return min(front, key=lambda s: s[1][0])
    if priority == "energy":
        return min(front, key=lambda s: s[1][1])
    t = [o[0] for _, o in front]
    e = [o[1] for _, o in front]

    def norm(x, lo, hi):
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    return min(front, key=lambda s: norm(s[1][0], min(t), max(t)) ** 2 + norm(s[1][1], min(e), max(e)) ** 2)


def sample_front(front, n):
    """Evenly spaced subset of the front (extremes included) to keep the LLM context short."""
    if n is None or len(front) <= n:
        return front
    step = (len(front) - 1) / (n - 1)
    return [front[round(i * step)] for i in range(n)]


def optimize_schedule(unavailable_machines=None, priority="balanced", seed=0, n_samples=6):
    """Tool entry point for the agent."""
    shop = load_shop()
    down = set(unavailable_machines or [])
    unknown = down - {m["id"] for m in shop["machines"]}
    if unknown:
        return {"error": f"unknown machines: {sorted(unknown)}"}
    machines = [m for m in shop["machines"] if m["id"] not in down]
    tasks = shop["tasks"]
    try:
        front = nsga2(tasks, machines, seed=seed)
    except ValueError as e:
        return {"error": str(e)}

    plan, (makespan, energy) = pick(front, priority)
    fastest = pick(front, "time")[1]
    greenest = pick(front, "energy")[1]
    assignment = {}
    for task, m in zip(tasks, plan):
        assignment.setdefault(machines[m]["id"], []).append(task["id"])
    return {
        "unavailable_machines": sorted(down),
        "priority": priority,
        "pareto_count": len(front),
        "fastest": {"makespan_h": round(fastest[0], 2), "energy_kwh": round(fastest[1], 1)},
        "lowest_energy": {"makespan_h": round(greenest[0], 2), "energy_kwh": round(greenest[1], 1)},
        "pareto_front": [{"makespan_h": round(o[0], 2), "energy_kwh": round(o[1], 1)}
                         for _, o in sample_front(front, n_samples)],
        "recommended": {
            "makespan_h": round(makespan, 2),
            "energy_kwh": round(energy, 1),
            "assignment": dict(sorted(assignment.items())),
        },
    }


def plot_front(results, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, r in results.items():
        f = r["pareto_front"]
        ax.plot([p["makespan_h"] for p in f], [p["energy_kwh"] for p in f], "o-", label=label, alpha=0.8)
        rec = r["recommended"]
        ax.scatter(rec["makespan_h"], rec["energy_kwh"], s=150, facecolors="none", edgecolors="red")
    ax.set_xlabel("Makespan (h)")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Pareto front (red circle = recommended)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    normal = optimize_schedule(n_samples=None)
    broken = optimize_schedule(unavailable_machines=["IM-2"], n_samples=None)
    for label, r in [("all machines", normal), ("IM-2 down", broken)]:
        rec = r["recommended"]
        print(f"{label}: {r['pareto_count']} Pareto plans, "
              f"recommended makespan {rec['makespan_h']} h / energy {rec['energy_kwh']} kWh")
        print("   ", rec["assignment"])
    plot_front({"all machines": normal, "IM-2 down": broken}, BASE / "pareto.png")
    print("saved pareto.png")


if __name__ == "__main__":
    main()
