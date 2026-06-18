from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


Individual = Dict[str, torch.Tensor]


def nd_sort(objectives: Sequence[Sequence[float]], n_select: int) -> List[int]:
    """Return indices from the first non-dominated fronts."""
    objs = np.asarray(objectives, dtype=float)
    n_points = len(objs)
    domination_count = np.zeros(n_points, dtype=int)
    dominated_solutions: List[List[int]] = [[] for _ in range(n_points)]
    fronts: List[List[int]] = [[]]

    for p in range(n_points):
        for q in range(n_points):
            if p == q:
                continue
            if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                dominated_solutions[p].append(q)
            elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    front_idx = 0
    while fronts[front_idx] and sum(len(front) for front in fronts) < n_select:
        next_front: List[int] = []
        for p in fronts[front_idx]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        front_idx += 1
        fronts.append(next_front)

    return [idx for front in fronts for idx in front][:n_select]


def individual_to_vector(individual: Individual) -> np.ndarray:
    arrays = [param.detach().cpu().numpy().reshape(-1) for param in individual.values()]
    return np.concatenate(arrays).astype(np.float32)


def vector_to_individual(vector: np.ndarray, template: Individual) -> Individual:
    new_individual: Individual = {}
    pointer = 0
    for name, param in template.items():
        numel = param.numel()
        values = vector[pointer : pointer + numel].reshape(tuple(param.shape))
        new_individual[name] = torch.tensor(values, dtype=param.dtype, device="cpu")
        pointer += numel
    return new_individual


def current_trainable_individual(model: torch.nn.Module) -> Individual:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def get_conflict_sites(gradients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    site1 = np.any(gradients < 0, axis=0) & np.any(gradients > 0, axis=0)
    site2 = np.all(np.abs(gradients) <= 1e-12, axis=0)
    return site1, site2


def multi_obj_two_point_gradients(
    trainer,
    model: torch.nn.Module,
    individual: Individual,
    objectives: Sequence[float],
    dataloader: Iterable[dict],
) -> tuple[np.ndarray, np.ndarray]:
    mu = getattr(trainer.args, "zo_eps", 1e-3)
    x_vec = individual_to_vector(individual)
    dimension = x_vec.size
    direction = np.random.randn(dimension).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-12

    perturbed = vector_to_individual(x_vec + mu * direction, individual)
    perturbed_objectives = trainer.evaluate_individual(model, perturbed, dataloader)

    gradients = []
    for obj_old, obj_new in zip(objectives, perturbed_objectives):
        gradients.append(dimension * (float(obj_new) - float(obj_old)) * direction / mu)
    return np.stack(gradients, axis=0), direction


def crowding_select(objectives: Sequence[Sequence[float]], candidates: List[int], n_select: int) -> List[int]:
    if len(candidates) <= n_select:
        return candidates

    objs = np.asarray([objectives[i] for i in candidates], dtype=float)
    n_points, n_obj = objs.shape
    distances = np.zeros(n_points, dtype=float)

    for obj_idx in range(n_obj):
        values = objs[:, obj_idx]
        order = np.argsort(values)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        denom = values[order[-1]] - values[order[0]]
        if abs(denom) < 1e-12:
            continue
        for rank in range(1, n_points - 1):
            distances[order[rank]] += (values[order[rank + 1]] - values[order[rank - 1]]) / denom

    keep = np.argsort(-distances)[:n_select]
    return [candidates[i] for i in keep]


def mozo_step(trainer, model: torch.nn.Module, inputs: Optional[dict]):
    """Run one MOZO update and write the best individual back to ``model``."""
    population_size = int(trainer.args.mozo_population_size)
    num_objectives = int(trainer.args.num_objectives)
    dataloader = [inputs] if inputs is not None else trainer.train_batch_dataloader

    if getattr(trainer, "weights", None) is None:
        trainer.weights = np.random.dirichlet(np.ones(num_objectives), size=population_size)

    if getattr(trainer, "mozo_pop", None) is None:
        template = current_trainable_individual(model)
        base_vec = individual_to_vector(template)
        trainer.mozo_template = template
        trainer.mozo_pop = []
        trainer.mozo_objs = []
        trainer.mozo_archive = []
        trainer.mozo_archive_objs = []
        trainer.K = [0] * population_size
        trainer.g0 = [None] * population_size
        trainer.d0 = [None] * population_size

        init_scale = getattr(trainer.args, "mozo_init_scale", 1e-3)
        for _ in range(population_size):
            vec = base_vec + init_scale * np.random.randn(base_vec.size).astype(np.float32)
            individual = vector_to_individual(vec, template)
            objectives = trainer.evaluate_individual(model, individual, dataloader)
            trainer.mozo_pop.append(individual)
            trainer.mozo_objs.append(objectives)
            trainer.mozo_archive.append(individual)
            trainer.mozo_archive_objs.append(objectives)

    offspring: List[Individual] = []
    offspring_objs: List[List[float]] = []
    step_size = getattr(trainer.args, "mozo_step_size", 1e-5)

    for idx in range(population_size):
        gradients, _ = multi_obj_two_point_gradients(
            trainer, model, trainer.mozo_pop[idx], trainer.mozo_objs[idx], dataloader
        )
        gk = gradients.T @ trainer.weights[idx]
        site1, site2 = get_conflict_sites(gradients)

        if trainer.K[idx] == 0 or trainer.g0[idx] is None or trainer.d0[idx] is None:
            dk = -gk
        else:
            beta = np.dot(gk, gk) / (np.dot(trainer.g0[idx], trainer.g0[idx]) + 1e-12)
            dk = -gk + beta * trainer.d0[idx]
            if np.dot(gk, dk) >= 0:
                dk = -gk
        trainer.K[idx] += 1

        pop_vec = individual_to_vector(trainer.mozo_pop[idx])
        a_idx, b_idx = np.random.choice(len(trainer.mozo_archive), 2, replace=True)
        a_vec = individual_to_vector(trainer.mozo_archive[a_idx])
        b_vec = individual_to_vector(trainer.mozo_archive[b_idx])

        delta = np.zeros_like(pop_vec)
        no_conflict = (~site2) & (~site1)
        mutation_prob = min(1.0, 1.0 / max(np.sum(site1), 1))
        mutation_mask = np.random.rand(pop_vec.size) < mutation_prob
        conflict = (site2 | site1) & mutation_mask
        delta[no_conflict] = step_size * dk[no_conflict]
        delta[conflict] = step_size * (a_vec[conflict] - b_vec[conflict])

        candidate = vector_to_individual(pop_vec + delta, trainer.mozo_template)
        candidate_obj = trainer.evaluate_individual(model, candidate, dataloader)
        offspring.append(candidate)
        offspring_objs.append(candidate_obj)

        if np.any(np.asarray(candidate_obj) < np.asarray(trainer.mozo_objs[idx])):
            trainer.mozo_pop[idx] = candidate
            trainer.mozo_objs[idx] = candidate_obj
            trainer.g0[idx] = gk
            trainer.d0[idx] = dk
        else:
            archive_idx = np.random.randint(len(trainer.mozo_archive))
            trainer.mozo_pop[idx] = trainer.mozo_archive[archive_idx]
            trainer.mozo_objs[idx] = trainer.mozo_archive_objs[archive_idx]
            trainer.K[idx] = 0

    combined = trainer.mozo_archive + offspring
    combined_objs = trainer.mozo_archive_objs + offspring_objs
    selected = nd_sort(combined_objs, population_size)
    selected = crowding_select(combined_objs, selected, population_size)
    trainer.mozo_archive = [combined[i] for i in selected]
    trainer.mozo_archive_objs = [combined_objs[i] for i in selected]

    best_idx = int(np.argmin([obj[0] for obj in trainer.mozo_archive_objs]))
    best_individual = trainer.mozo_archive[best_idx]
    trainer.load_individual(model, best_individual)
    return trainer.mozo_archive_objs[best_idx]
