"""Compare three optimizers under the same FP-TBF-DI replay protocol."""
import argparse
import json
from functools import partial

import numpy as np
from threadpoolctl import threadpool_limits

from run_paper_pso_fusion import ROOT, initial_weights, make_fitness, pso, simplex, study

SEEDS = (20261117, 20261118, 20261119, 20261120, 20261121)


def genetic(scores, labels, seed):
    rng = np.random.default_rng(seed)
    fitness = make_fitness(scores, labels)
    population = initial_weights(rng)
    values = fitness(population)
    size = len(population)
    for _ in range(100):
        contestants = rng.integers(size, size=(size, 2, 3))
        winners = values[contestants].argmax(axis=2)
        parents = contestants[np.arange(size)[:, None], np.arange(2), winners]
        mix = rng.random((size, 1))
        children = mix * population[parents[:, 0]] + (1 - mix) * population[parents[:, 1]]
        mutate = rng.random(size) < 0.5
        scales = rng.choice([0.015, 0.08], size=(mutate.sum(), 1))
        children[mutate] += scales * rng.normal(size=(mutate.sum(), 3))
        children = simplex(children)
        best = values.argmax()
        elite, elite_value = population[best].copy(), values[best]
        population, values = children, fitness(children)
        worst = values.argmin()
        population[worst], values[worst] = elite, elite_value
    return population[values.argmax()]


def bee_colony(scores, labels, seed):
    rng = np.random.default_rng(seed)
    fitness = make_fitness(scores, labels)
    initial = initial_weights(rng)
    initial_values = fitness(initial)
    order = np.argsort(-initial_values, kind='stable')[:48]
    sources, values = initial[order], initial_values[order]
    size = len(sources)
    indices = np.arange(size)
    stale = np.zeros(size, dtype=int)
    best, best_value = sources[0].copy(), values[0]

    def neighbors(target):
        peer = (target + rng.integers(1, size, len(target))) % size
        coordinate = rng.integers(3, size=len(target))
        proposal = sources[target].copy()
        row = np.arange(len(target))
        proposal[row, coordinate] += rng.uniform(-1, 1, len(target)) * (
            sources[target, coordinate] - sources[peer, coordinate])
        return simplex(proposal)

    for _ in range(100):
        # Replace stalled sources within the employed phase's evaluation budget.
        scouts = stale >= 15
        proposals = neighbors(indices)
        proposals[scouts] = rng.dirichlet(np.ones(3), size=scouts.sum())
        proposed_values = fitness(proposals)
        accept = scouts | (proposed_values > values)
        sources[accept], values[accept] = proposals[accept], proposed_values[accept]
        stale[accept] = 0
        stale[~accept] += 1
        current = values.argmax()
        if values[current] > best_value:
            best, best_value = sources[current].copy(), values[current]

        # Onlookers select sources by fitness rank; all proposals use the same phase snapshot.
        ranks = np.empty(size)
        ranks[np.argsort(values, kind='stable')] = np.arange(1, size + 1)
        targets = rng.choice(indices, size=size, p=ranks / ranks.sum())
        candidates = neighbors(targets)
        candidate_values = fitness(candidates)
        for target, candidate, value in zip(targets, candidates, candidate_values, strict=True):
            if value > values[target]:
                sources[target], values[target], stale[target] = candidate, value, 0
            else:
                stale[target] += 1
        current = values.argmax()
        if values[current] > best_value:
            best, best_value = sources[current].copy(), values[current]
    return best


def ant_colony(scores, labels, seed):
    """ACOR with Euclidean projection onto the fusion-weight simplex."""
    rng = np.random.default_rng(seed)
    fitness = make_fitness(scores, labels)
    archive = initial_weights(rng)
    values = fitness(archive)
    size = len(archive)
    order = np.argsort(-values, kind='stable')
    archive, values = archive[order], values[order]
    rank_weights = np.exp(-0.5 * (np.arange(size) / (0.1 * size))**2)
    probabilities = rank_weights / rank_weights.sum()
    for _ in range(100):
        selected = rng.choice(size, size=size, p=probabilities)
        bandwidth = 0.85 * np.abs(archive[:, None] - archive[None, :]).sum(axis=1) / (size - 1)
        proposals = simplex(rng.normal(archive[selected], bandwidth[selected]))
        candidates = np.vstack((archive, proposals))
        candidate_values = np.concatenate((values, fitness(proposals)))
        keep = np.argsort(-candidate_values, kind='stable')[:size]
        archive, values = candidates[keep], candidate_values[keep]
    return archive[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-acor', action='store_true', help='Add ACOR to the existing comparison')
    args = parser.parse_args()
    output = ROOT / 'results/paper_fusion_optimizer_comparison'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'results.json'
    algorithms = [('PSO', pso), ('GA', genetic), ('ABC', bee_colony), ('ACOR', ant_colony)]
    if args.only_acor:
        results = json.loads(path.read_text(encoding='utf-8'))
        for name, _ in algorithms[:-1]:
            if set(results[name]) != {str(seed) for seed in SEEDS}:
                raise ValueError(f'{name} results do not match the required search seeds.')
        algorithms = algorithms[-1:]
    else:
        results = {}
    for name, optimizer in algorithms:
        results[name] = {}
        for seed in SEEDS:
            result = study(partial(optimizer, seed=seed))
            results[name][seed] = result
            stream = sum(r['pso']['correct'] for r in result['prequential'])
            holdout = sum(r['pso']['correct'] for r in result['retrospective_holdout'])
            apparent = result['pooled_in_sample']['pso']['correct']
            print(f'{name} {seed}: next batch={stream}, holdout={holdout}, apparent={apparent}', flush=True)
    path.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
