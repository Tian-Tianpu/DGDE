"""NGDE for minimization, YQ2024 Algorithms 1--3 and Eqs. (2)--(6).

API: NGDE(NP, dimension, niche_size, budget, F, fitness_p, learning_rate)
     model.NGDE(lower, upper, objective, gradient)
NumPy only; pop[*].solution/target.fitness and g_best are compatible.

Explicit interpretations:
* k increments after ALL niches (generation definition and Fig. 2).
* selection_mode='paper_printed': Algorithm 1 line 14 is followed literally:
  on parent survival, available = original_niche minus that parent only.
  'cumulative' instead removes survivors cumulatively; this is an optional
  interpretation, NOT a claim that the printed line says so.
* Equal niche objective values use p=0 (Eq. 3 otherwise divides by zero).
* Donors come from the full frozen parent population; two distinct indices
  by default. The paper does not explicitly specify replacement/exclusion.
* Project complete offspring to the box as stated on p.6. Gradients are at x,
  not at a differential trial point (that would be a different algorithm).
"""
from types import SimpleNamespace
import numpy as np


class NGDE:
    def __init__(self, population_size, dimension, niche_size, budget, F,
                 fitness_p, learning_rate, *, selection_mode='paper_printed',
                 store_population_history=True, donor_replace=False, seed=None):
        for name, value, minimum in [('population_size', population_size, 2),
                ('dimension', dimension, 1), ('niche_size', niche_size, 1), ('budget', budget, 0)]:
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f'{name} must be integer >= {minimum}')
        if not np.isfinite(F) or F <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('F and learning_rate must be finite and positive')
        if not 0 < fitness_p < 1:
            raise ValueError('fitness_p must be in (0,1)')
        if selection_mode not in ('paper_printed', 'cumulative'):
            raise ValueError('Unknown selection_mode')
        self.population_size, self.dimension = population_size, dimension
        self.niche_size, self.budget = niche_size, budget
        self.F, self.fitness_p, self.learning_rate = F, fitness_p, learning_rate
        self.selection_mode = selection_mode
        self.store_population_history = store_population_history
        self.donor_replace = donor_replace
        self.seed = seed
        self.rng = np.random if seed is None else np.random.RandomState(seed)
        self._reset()

    def _reset(self):
        self.t = self.nfe = self.nge = self.step_projections = 0
        self.history, self.best_history = [], []
        self.population_history, self.population_all = [], []
        self.niche_history, self.niche_info_history = [], []
        self.pop, self.g_best, self.best_so_far = [], None, None
        self.stop_reason = 'not_started'

    @staticmethod
    def p_distance(x, y):
        return float(np.linalg.norm(np.asarray(x)-np.asarray(y)))

    def initial(self, x_min, x_max):
        return self.rng.uniform(x_min, x_max, (self.population_size, self.dimension))

    def _fitness(self, x, f):
        self.nfe += 1
        value = float(f(np.array(x, copy=True)))
        if not np.isfinite(value):
            raise ValueError('Nonfinite objective value')
        return value

    def Fitness(self, population, f):
        return np.array([self._fitness(x, f) for x in population])

    def Gradient(self, x, f_gradient):
        """Return gradient, never change x."""
        self.nge += 1
        grad = np.asarray(f_gradient(np.array(x, copy=True)), dtype=float)
        if grad.shape != np.shape(x) or not np.all(np.isfinite(grad)):
            raise ValueError('Gradient must have the shape of x and be finite')
        return grad

    def update1(self, x, f_gradient):
        return np.asarray(x)-self.learning_rate*self.Gradient(x, f_gradient)

    def update2(self, x, best_x, f_gradient):
        return (np.asarray(x)-self.learning_rate*self.Gradient(x, f_gradient)
                + self.F*(np.asarray(best_x)-np.asarray(x)))

    def update3(self, x, population):
        a, b = self.rng.choice(len(population), 2, replace=self.donor_replace)
        return np.asarray(x)+self.F*(population[a]-population[b])

    def is_domin(self, x, x_min, x_max):
        if not np.all(np.isfinite(x)):
            raise ValueError('Nonfinite offspring before projection')
        clipped = np.clip(x, x_min, x_max)
        self.step_projections += int(np.any(clipped != x))
        return clipped

    @staticmethod
    def _normalized(values):
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi == lo:
            return np.zeros(len(values))
        # Scaling avoids overflow of hi-lo for finite objective values.
        scale = max(abs(lo), abs(hi), 1.0)
        return (values/scale-lo/scale)/(hi/scale-lo/scale)

    def _niche_indices(self, population, fitness):
        remaining = list(range(len(population)))
        niches = []
        while remaining:
            center = min(remaining, key=lambda i: fitness[i])
            others = [i for i in remaining if i != center]
            others.sort(key=lambda i: self.p_distance(population[i], population[center]))
            niche = [center]+others[:self.niche_size-1]
            niches.append(niche)
            selected = set(niche)
            remaining = [i for i in remaining if i not in selected]
        return niches

    def get_niche(self, population, f):
        population = np.array(population, dtype=float, copy=True)
        return [population[ids].copy() for ids in self._niche_indices(population, self.Fitness(population, f))]

    def _select(self, candidate, value, parents, fitness, original, available):
        nearest = min(available, key=lambda i: self.p_distance(candidate, parents[i]))
        if value <= fitness[nearest]:
            return candidate.copy(), value, available
        if self.selection_mode == 'paper_printed':
            available = [i for i in original if i != nearest]
        else:
            available = [i for i in available if i != nearest]
        return parents[nearest].copy(), float(fitness[nearest]), available

    @staticmethod
    def _agent(x, value):
        return SimpleNamespace(solution=x.copy(), target=SimpleNamespace(fitness=float(value)))

    def NGDE(self, x_min, x_max, f, f_gradient):
        self._reset()
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)
        lower = np.broadcast_to(np.asarray(x_min, dtype=float), (self.dimension,))
        upper = np.broadcast_to(np.asarray(x_max, dtype=float), (self.dimension,))
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower >= upper):
            raise ValueError('Require finite lower < upper for every dimension')
        parents = self.initial(lower, upper)
        fitness = self.Fitness(parents, f)
        best = int(np.argmin(fitness))
        self.best_so_far = self._agent(parents[best], fitness[best])
        self.history = [float(fitness[best])]
        for _ in range(self.budget):
            # All candidate mutations read the unchanged generation P^k.
            niches = self._niche_indices(parents, fitness)
            children, child_fitness, info = [], [], []
            for niche_index, ids in enumerate(niches):
                best_index = min(ids, key=lambda i: fitness[i])
                probabilities = self._normalized(fitness[ids])
                available = list(ids)
                if self.store_population_history:
                    info.append(dict(niche_index=niche_index, parent_indices=list(ids),
                                     individuals=parents[ids].tolist()))
                for index, probability in zip(ids, probabilities):
                    if probability > self.fitness_p:
                        candidate = self.update3(parents[index], parents)
                    else:
                        candidate = self.update2(parents[index], parents[best_index], f_gradient)
                    candidate = self.is_domin(candidate, lower, upper)
                    value = self._fitness(candidate, f)
                    child, value, available = self._select(candidate, value, parents, fitness, ids, available)
                    children.append(child)
                    child_fitness.append(value)
            parents, fitness = np.array(children), np.array(child_fitness)
            self.t += 1
            best = int(np.argmin(fitness))
            if fitness[best] < self.best_so_far.target.fitness:
                self.best_so_far = self._agent(parents[best], fitness[best])
            self.history.append(float(fitness[best]))
            self.best_history.append(self.best_so_far.target.fitness)
            self.niche_history.append(len(niches))
            if self.store_population_history:
                self.population_history.append([(x.tolist(), float(v)) for x, v in zip(parents, fitness)])
                self.population_all.append(parents.copy())
                self.niche_info_history.append(info)
        self.pop = [self._agent(x, v) for x, v in zip(parents, fitness)]
        self.g_best = min(self.pop, key=lambda a: a.target.fitness)
        self.stop_reason = 'generations_completed'
        return self.g_best
