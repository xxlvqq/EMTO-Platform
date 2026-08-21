# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2026/7/14 下午8:02
# @Author: xxx
# @Introduction: ATCT — Adaptive Task-Centric Transfer for Evolutionary Multitask Optimization
# @Description:

from collections import deque

import numpy as np

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.Operator.DE_operator.DE import DE
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem

class LSHADE_Memory:
    """Per-task LSHADE parameter memory."""

    def __init__(self, H=6, N_init=100, archive_factor=1.4):
        self.H = H
        self.N_init = N_init
        self.N_min = 4
        self.M_F = deque([0.5] * H, maxlen=H)
        self.M_CR = deque([0.5] * H, maxlen=H)
        self.archive = deque(maxlen=int(N_init * archive_factor))
        self.N_current = N_init
        self.k = 0                        # memory write pointer
        self.gen = 0

    def generate_f_cr(self, n):
        """Sample F ~ Cauchy(M_F[ri], 0.1),  CR ~ Normal(M_CR[ri], 0.1)."""
        F = np.zeros(n)
        CR = np.zeros(n)
        for i in range(n):
            ri = np.random.randint(0, self.H)
            CR[i] = np.clip(np.random.normal(self.M_CR[ri], 0.1), 0.0, 1.0)
            f_val = np.random.standard_cauchy() * 0.1 + self.M_F[ri]
            while f_val <= 0.0:
                f_val = np.random.standard_cauchy() * 0.1 + self.M_F[ri]
            F[i] = min(f_val, 1.0)
        return F, CR

    def update_memory(self, parents, offspring, F, CR, mask=None):
        """Weighted update: Lehmer mean for F, arithmetic mean for CR.

        `mask`: optional boolean array selecting which individuals to consider
        (e.g., only those that used the LSHADE operator).  Default None = all.
        """
        S_F, S_CR, w = [], [], []
        for i in range(len(offspring)):
            if mask is not None and not mask[i]:
                continue
            if offspring[i].obj < parents[i].obj:
                self.archive.append(parents[i])
                delta = parents[i].obj - offspring[i].obj
                w.append(delta)
                S_F.append(F[i])
                S_CR.append(CR[i])

        if len(S_F) > 0:
            w = np.array(w, dtype=float)
            w_sum = w.sum()
            if w_sum > 1e-12:
                w = w / w_sum
            else:
                w = np.ones_like(w) / len(w)
            self.M_F[self.k] = np.sum(w * np.array(S_F) ** 2) / (np.sum(w * np.array(S_F)) + 1e-12)
            self.M_CR[self.k] = np.sum(w * np.array(S_CR))
            self.k = (self.k + 1) % self.H

    def reduce_population(self, population, N_next):
        """Remove the worst individuals during LPSR."""
        if N_next >= self.N_current:
            return population, self.N_current
        n_keep = max(self.N_min, N_next)
        # Keep best n_keep
        idx = np.argsort([ind.obj for ind in population])
        new_pop = [population[i] for i in idx[:n_keep]]
        return new_pop, n_keep

    def lpsr_target(self, FE, maxFE):
        """Compute the next population size by linear reduction."""
        if maxFE <= 1:
            return self.N_min
        ratio = 1.0 - min(1.0, FE / maxFE)
        return max(self.N_min,
                   int(round(self.N_min + (self.N_init - self.N_min) * ratio)))


# ------------------------------------------------------------------
#  ATCTMTO algorithm
# ------------------------------------------------------------------

class ATCTMTO(Algorithm):
    """
    Adaptive Task-Centric Transfer for Multitask Optimization.(ATCTMTO)

    - Each task maintains an independent sub-population with Linear Population
      Size Reduction (LPSR, N_init → N_min).
    - Task-internal: adaptive differential evolution operator
      (current-to-pbest/1 with SHADE-style memory H=6).
    - Inter-task: centroid-aligned knowledge pool from elite individuals
      of source and target tasks.
    - Transfer probability adapted online by transfer success rate.
    """

    def __init__(self):
        super().__init__()
        self.mem = None
        self.elite_rate = 0.2
        self.knowledge_rate = 0.2
        self.p_transfer_min = 0.05
        self.p_transfer_max = 0.30
        self.noise_scale = 0.05
        self.transfer_success = None
        self.transfer_trials = None

    # ==================================================================
    #  Auxiliary methods
    # ==================================================================

    def _success_rate(self, t, s):
        """Online transfer success rate from task s to task t."""
        r = self.transfer_success[t][s] / (self.transfer_trials[t][s] + 1e-12)
        return float(np.clip(r, 0.0, 1.0))

    def _transfer_probability(self, t, s):
        """Compute the adaptive transfer probability."""
        sr = self._success_rate(t, s)
        lo, hi = self.p_transfer_min, self.p_transfer_max
        return float(lo + (hi - lo) * sr)

    def _select_source_task(self, t):
        """Select a source task s != t for target task t."""
        if Problem.T == 2:
            return 1 - t
        candidates = [s for s in range(Problem.T) if s != t]
        rates = np.array([self._success_rate(t, s) for s in candidates])
        if rates.sum() < 1e-12:
            return np.random.choice(candidates)
        probs = rates / rates.sum()
        return np.random.choice(candidates, p=probs)

    def _top_elites(self, pop, rate):
        """Return top `max(2, int(len(pop)*rate))` individuals sorted by obj."""
        n = max(2, int(len(pop) * rate))
        idx = np.argsort([ind.obj for ind in pop])
        return [pop[i] for i in idx[:n]]

    # ==================================================================
    #  Centroid-aligned knowledge pool
    # ==================================================================

    def _align_dimensions(self, x_rnvec, Cs, Ct, source_dim, target_dim):
        """Randomly align source dimensions with the target dimensions."""
        ds, dt = source_dim, target_dim
        if ds == dt:
            indices = np.arange(ds)
        else:
            indices = np.random.choice(ds, dt, replace=(dt > ds))
        return x_rnvec[indices] - Cs[indices] + Ct

    def _build_knowledge_pool(self, population, t, s, Prob):
        """
        Build centroid-aligned knowledge pool for source→target transfer.

        1. Compute centroids Ct, Cs of target and source elites.
        2. Compute target elite std for noise injection.
        3. For each pool member: sample source elite x,
           map  x' = x - Cs + Ct + noise,
           clip to [0,1].

        No evaluation is performed on pool individuals.
        """
        target_elites = self._top_elites(population[t], self.elite_rate)
        source_elites = self._top_elites(population[s], self.elite_rate)

        target_dim = Prob[t].dim
        source_dim = Prob[s].dim

        Ct = np.mean([ind.rnvec for ind in target_elites], axis=0)
        Cs = np.mean([ind.rnvec for ind in source_elites], axis=0)
        target_std = np.std([ind.rnvec for ind in target_elites], axis=0) + 1e-12

        pool_size = max(1, int(self.mem[t].N_current * self.knowledge_rate))
        knowledge_pool = []

        for _ in range(pool_size):
            x = source_elites[np.random.randint(len(source_elites))]
            # Random dimension alignment and centroid alignment.
            mapped = self._align_dimensions(
                x.rnvec, Cs, Ct, source_dim, target_dim)
            noise = self.noise_scale * target_std * np.random.randn(target_dim)
            mapped = np.clip(mapped + noise, 0.0, 1.0)

            ind = Individual()
            ind.rnvec = mapped
            knowledge_pool.append(ind)

        return knowledge_pool

    # ==================================================================
    #  Offspring generation  (adaptive DE + knowledge transfer)
    # ==================================================================

    def _generate_offspring(self, population, t, s, knowledge_pool):
        """Generate offspring with LSHADE and knowledge transfer."""
        N_cur = self.mem[t].N_current
        F_shade, CR_shade = self.mem[t].generate_f_cr(N_cur)

        F = np.zeros(N_cur)
        CR = np.zeros(N_cur)
        lshade_mask = np.ones(N_cur, dtype=bool)

        offspring = [Individual() for _ in range(N_cur)]
        transfer_flags = np.zeros(N_cur, dtype=bool)

        p_transfer = self._transfer_probability(t, s)
        obj_sorted_idx = np.argsort([ind.obj for ind in population[t]])
        archive = self.mem[t].archive

        for i in range(N_cur):
            parent = population[t][i]
            do_transfer = bool(knowledge_pool and np.random.rand() < p_transfer)
            p_rate = np.random.uniform(2.0 / N_cur, 0.2)
            n_elite = max(2, int(N_cur * p_rate))
            elite_idx = obj_sorted_idx[:n_elite]
            if do_transfer:
                candidates = [population[t][idx] for idx in elite_idx] + knowledge_pool
                pbest = candidates[np.random.randint(len(candidates))]
                transfer_flags[i] = pbest in knowledge_pool
            else:
                pbest = population[t][elite_idx[np.random.randint(len(elite_idx))]]

            r1_idx = int(np.random.choice([j for j in range(N_cur) if j != i]))
            total_pool = list(range(N_cur)) + list(range(N_cur, N_cur + len(archive)))
            valid = [j for j in total_pool if j != i and j != r1_idx]
            r2_idx = int(np.random.choice(valid))
            r1 = population[t][r1_idx]
            r2 = archive[r2_idx - N_cur] if r2_idx >= N_cur else population[t][r2_idx]
            mutant = (parent.rnvec
                      + F_shade[i] * (pbest.rnvec - parent.rnvec)
                      + F_shade[i] * (r1.rnvec - r2.rnvec))
            F[i], CR[i] = F_shade[i], CR_shade[i]
            trial = DE(parent.rnvec, mutant, len(parent.rnvec), CR[i])
            offspring[i].rnvec = np.clip(trial, 0.0, 1.0)

        return offspring, transfer_flags, F, CR, lshade_mask

    # ==================================================================
    #  Transfer statistics
    # ==================================================================

    def _update_transfer_statistics(self, parents, offspring, transfer_flags, t, s):
        """Accumulate transfer success / trial counts."""
        for i in range(len(offspring)):
            if transfer_flags[i]:
                self.transfer_trials[t][s] += 1
                if offspring[i].obj < parents[i].obj:
                    self.transfer_success[t][s] += 1

    # ==================================================================
    #  Pairwise replacement  (LSHADE-style: offspring wins → replace parent)
    # ==================================================================

    @staticmethod
    def _pairwise_replace(parents, offspring):
        """Standard LSHADE pairwise replacement: offspring replaces parent if better."""
        new_pop = []
        for i in range(len(offspring)):
            if offspring[i].obj <= parents[i].obj:
                new_pop.append(offspring[i])
            else:
                new_pop.append(parents[i])
        return new_pop

    # ==================================================================
    #  Main loop
    # ==================================================================

    def run(self, Prob, isPrint=False):
        N_init = 100
        Problem.N = N_init

        # ---- Initialise sub-populations ----
        population = Initialization(self, Prob, Individual, False)

        # ---- Per-task LSHADE memory ----
        self.mem = [LSHADE_Memory(H=6, N_init=N_init) for _ in range(Problem.T)]

        # ---- Transfer statistics (T x T, diagonal unused) ----
        self.transfer_success = np.ones((Problem.T, Problem.T))
        self.transfer_trials = np.ones((Problem.T, Problem.T)) * 5

        while self.notTerminated(Prob, isPrint):
            for t in range(Problem.T):
                # Update the population size.
                N_next = self.mem[t].lpsr_target(self.FE, Problem.maxFE)
                if N_next < self.mem[t].N_current:
                    population[t], self.mem[t].N_current = \
                        self.mem[t].reduce_population(population[t], N_next)

                N_cur = self.mem[t].N_current

                # 1. Select source task
                s = self._select_source_task(t)

                # Build the knowledge pool.
                knowledge_pool = self._build_knowledge_pool(population, t, s, Prob)

                # Generate and evaluate offspring.
                offspring, transfer_flags, F, CR, lshade_mask = \
                    self._generate_offspring(population, t, s, knowledge_pool)

                self.Evaluation(offspring, Prob[t], t)

                self._update_transfer_statistics(
                    population[t], offspring, transfer_flags, t, s)

                self.mem[t].update_memory(
                    population[t], offspring, F, CR, lshade_mask)

                population[t] = self._pairwise_replace(population[t], offspring)

        return self


# ------------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------------

def main():
    # ---- CEC17 MTSO benchmarks ----
    Prob = CI_HS()
    # Prob = CI_MS()
    # Prob = CI_LS()
    # Prob = PI_HS()
    # Prob = PI_MS()
    # Prob = PI_LS()
    # Prob = NI_HS()
    # Prob = NI_MS()
    # Prob = NI_LS()

    # ---- WCCI20 MTSO benchmarks ----
    # Prob = Benchmark1()
    # Prob = Benchmark2()
    # Prob = Benchmark3()
    # Prob = Benchmark4()
    # Prob = Benchmark5()
    # Prob = Benchmark6()
    # Prob = Benchmark7()
    # Prob = Benchmark8()
    # Prob = Benchmark9()
    # Prob = Benchmark10()

    repeat = 1
    Problem.maxFE = 100000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = ATCTMTO().run(Prob, True)
        costs[i] = result.Best
        print(f'Repetition {i} :', result.Best, '\n')

        print(f'Values of the previous {i + 1} generations:')
        for j in range(i + 1):
            print(*costs[j])

        print(f'Average Values of the previous {i + 1} generations:')
        print(np.mean(costs[:i + 1], axis=0))
        print()


if __name__ == "__main__":
    main()
