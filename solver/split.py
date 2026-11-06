"""Prins Split algorithm for CVRPTW.

Partitions a continuous random-key giant tour into optimal feasible vehicle
routes using a shortest-path-on-DAG formulation (Prins, 2004).
Includes time windows with waiting times, service times, driver duty limits,
priority customer scaling, and adaptive constraint penalties.
Supports Numba JIT acceleration with pure NumPy fallback.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np

# Optional Numba JIT acceleration
try:
    from numba import njit  # type: ignore
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Fallback no-op decorator when numba is not installed."""
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        def decorator(func: Any) -> Any:
            return func
        return decorator

from solver.models import Customer, Problem, Route
from solver.tensor import CostTensor, precompute_cost_tensor


@njit(fastmath=True)
def _split_cost_numba(
    tour: np.ndarray,
    cost_data: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    max_vehicles: int = -1,
    overflow_penalty: float = 10000.0,
    t_start: float = 0.0,
    bucket_size: float = 900.0,
    service_times: np.ndarray | None = None,
    ready_times: np.ndarray | None = None,
    due_times: np.ndarray | None = None,
    depot_due_time: float = 1e12,
    priority_factors: np.ndarray | None = None,
    max_duty: float = -1.0,
    alpha: float = 100.0,
    beta: float = 100.0,
    gamma: float = 100.0,
    hard_time_windows: bool = False,
    hard_duty_limit: bool = False,
) -> float:
    """Numba-accelerated evaluation of the optimal split cost for a giant tour.

    Supports pure CVRP (fast O(1) stationary loop) as well as CVRPTW with
    waiting times, service times, driver duty limits, and priority lateness penalties.
    """
    n = len(tour)
    if n == 0:
        return 0.0

    is_stationary = (cost_data.ndim == 2) or (cost_data.shape[0] == 1)
    has_services = service_times is not None
    has_priorities = priority_factors is not None

    # Check if time windows are active (any customer due_time is constrained)
    has_tw = False
    if due_times is not None:
        for idx in range(len(due_times)):
            if due_times[idx] < 1e11:
                has_tw = True
                break

    # ------------------------------------------------------------------
    # 1. Stationary Path (2D Matrix or B=1 Tensor)
    # ------------------------------------------------------------------
    if is_stationary:
        dist_matrix = cost_data if cost_data.ndim == 2 else cost_data[0]

        # Case A: Pure CVRP without time windows & duty limits (O(1) incremental update)
        if not has_tw and max_duty <= 0.0:
            if max_vehicles <= 0:
                V = np.full(n + 1, 1e12, dtype=np.float64)
                V[0] = 0.0

                for i in range(n):
                    if V[i] >= 1e12:
                        continue
                    load = 0.0
                    cost = 0.0

                    for j in range(i + 1, n + 1):
                        cust = tour[j - 1]
                        load += demands[cust]
                        if load > capacity:
                            break

                        if j == i + 1:
                            cost = dist_matrix[0, cust] + dist_matrix[cust, 0]
                        else:
                            prev = tour[j - 2]
                            cost = cost - dist_matrix[prev, 0] + dist_matrix[prev, cust] + dist_matrix[cust, 0]

                        cand = V[i] + cost
                        if cand < V[j]:
                            V[j] = cand

                return V[n]

            # Constrained vehicle count: 2D DP table V[j, v]
            K = max_vehicles
            V_mat = np.full((n + 1, K + 1), 1e12, dtype=np.float64)
            V_mat[0, 0] = 0.0

            for v in range(1, K + 1):
                for i in range(n):
                    if V_mat[i, v - 1] >= 1e12:
                        continue
                    load = 0.0
                    cost = 0.0

                    for j in range(i + 1, n + 1):
                        cust = tour[j - 1]
                        load += demands[cust]
                        if load > capacity:
                            break

                        if j == i + 1:
                            cost = dist_matrix[0, cust] + dist_matrix[cust, 0]
                        else:
                            prev = tour[j - 2]
                            cost = cost - dist_matrix[prev, 0] + dist_matrix[prev, cust] + dist_matrix[cust, 0]

                        cand = V_mat[i, v - 1] + cost
                        if cand < V_mat[j, v]:
                            V_mat[j, v] = cand

            min_cost = 1e12
            for v in range(1, K + 1):
                if V_mat[n, v] < min_cost:
                    min_cost = V_mat[n, v]

            if min_cost < 1e12:
                return min_cost

            unc_cost = _split_cost_numba(
                tour, cost_data, demands, capacity, max_vehicles=-1, overflow_penalty=overflow_penalty
            )
            return unc_cost + overflow_penalty

        # Case B: Stationary CVRPTW (time windows / duty limits active)
        if max_vehicles <= 0:
            V = np.full(n + 1, 1e12, dtype=np.float64)
            V[0] = 0.0

            for i in range(n):
                if V[i] >= 1e12:
                    continue
                load = 0.0
                t = t_start
                prev = 0
                route_dist = 0.0
                tw_viol = 0.0

                for j in range(i + 1, n + 1):
                    cust = tour[j - 1]
                    load += demands[cust]
                    if load > capacity:
                        break

                    travel = dist_matrix[prev, cust]
                    route_dist += travel
                    arr = t + travel

                    if has_tw and ready_times is not None and due_times is not None:
                        if arr < ready_times[cust]:
                            start_serv = ready_times[cust]
                        else:
                            start_serv = arr
                            if arr > due_times[cust]:
                                lateness = arr - due_times[cust]
                                if hard_time_windows:
                                    break
                                p_fac = priority_factors[cust] if has_priorities else 1.0
                                tw_viol += lateness * p_fac
                    else:
                        start_serv = arr

                    serv = service_times[cust] if has_services else 0.0
                    t = start_serv + serv
                    prev = cust

                    # Return to depot from cust
                    ret_travel = dist_matrix[cust, 0]
                    total_dist = route_dist + ret_travel
                    depot_arr = t + ret_travel

                    depot_late = 0.0
                    if has_tw and depot_arr > depot_due_time:
                        depot_late = depot_arr - depot_due_time
                        if hard_time_windows:
                            continue

                    route_dur = depot_arr - t_start
                    duty_viol = 0.0
                    if max_duty > 0.0 and route_dur > max_duty:
                        duty_viol = route_dur - max_duty
                        if hard_duty_limit:
                            continue

                    arc_cost = total_dist + beta * (tw_viol + depot_late) + gamma * duty_viol
                    cand = V[i] + arc_cost
                    if cand < V[j]:
                        V[j] = cand

            return V[n]

        # Stationary CVRPTW with vehicle limit
        K = max_vehicles
        V_mat = np.full((n + 1, K + 1), 1e12, dtype=np.float64)
        V_mat[0, 0] = 0.0

        for v in range(1, K + 1):
            for i in range(n):
                if V_mat[i, v - 1] >= 1e12:
                    continue
                load = 0.0
                t = t_start
                prev = 0
                route_dist = 0.0
                tw_viol = 0.0

                for j in range(i + 1, n + 1):
                    cust = tour[j - 1]
                    load += demands[cust]
                    if load > capacity:
                        break

                    travel = dist_matrix[prev, cust]
                    route_dist += travel
                    arr = t + travel

                    if has_tw and ready_times is not None and due_times is not None:
                        if arr < ready_times[cust]:
                            start_serv = ready_times[cust]
                        else:
                            start_serv = arr
                            if arr > due_times[cust]:
                                lateness = arr - due_times[cust]
                                if hard_time_windows:
                                    break
                                p_fac = priority_factors[cust] if has_priorities else 1.0
                                tw_viol += lateness * p_fac
                    else:
                        start_serv = arr

                    serv = service_times[cust] if has_services else 0.0
                    t = start_serv + serv
                    prev = cust

                    ret_travel = dist_matrix[cust, 0]
                    total_dist = route_dist + ret_travel
                    depot_arr = t + ret_travel

                    depot_late = 0.0
                    if has_tw and depot_arr > depot_due_time:
                        depot_late = depot_arr - depot_due_time
                        if hard_time_windows:
                            continue

                    route_dur = depot_arr - t_start
                    duty_viol = 0.0
                    if max_duty > 0.0 and route_dur > max_duty:
                        duty_viol = route_dur - max_duty
                        if hard_duty_limit:
                            continue

                    arc_cost = total_dist + beta * (tw_viol + depot_late) + gamma * duty_viol
                    cand = V_mat[i, v - 1] + arc_cost
                    if cand < V_mat[j, v]:
                        V_mat[j, v] = cand

        min_cost = 1e12
        for v in range(1, K + 1):
            if V_mat[n, v] < min_cost:
                min_cost = V_mat[n, v]

        if min_cost < 1e12:
            return min_cost

        unc_cost = _split_cost_numba(
            tour, cost_data, demands, capacity, max_vehicles=-1,
            overflow_penalty=overflow_penalty, t_start=t_start,
            bucket_size=bucket_size, service_times=service_times,
            ready_times=ready_times, due_times=due_times,
            depot_due_time=depot_due_time, priority_factors=priority_factors,
            max_duty=max_duty, alpha=alpha, beta=beta, gamma=gamma,
            hard_time_windows=hard_time_windows, hard_duty_limit=hard_duty_limit,
        )
        return unc_cost + overflow_penalty

    # ------------------------------------------------------------------
    # 2. Time-Dependent Tensor Path (B > 1, e.g. Traffic / Valhalla)
    # ------------------------------------------------------------------
    B = cost_data.shape[0]

    if max_vehicles <= 0:
        V = np.full(n + 1, 1e12, dtype=np.float64)
        V[0] = 0.0

        for i in range(n):
            if V[i] >= 1e12:
                continue
            load = 0.0
            t = t_start
            prev = 0
            tw_viol = 0.0

            for j in range(i + 1, n + 1):
                cust = tour[j - 1]
                load += demands[cust]
                if load > capacity:
                    break

                # Hop from prev to cust
                b = int((t - t_start) / bucket_size)
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1

                travel = cost_data[b, prev, cust]
                arr = t + travel

                if has_tw and ready_times is not None and due_times is not None:
                    if arr < ready_times[cust]:
                        start_serv = ready_times[cust]
                    else:
                        start_serv = arr
                        if arr > due_times[cust]:
                            lateness = arr - due_times[cust]
                            if hard_time_windows:
                                break
                            p_fac = priority_factors[cust] if has_priorities else 1.0
                            tw_viol += lateness * p_fac
                else:
                    start_serv = arr

                serv = service_times[cust] if has_services else 0.0
                t = start_serv + serv
                prev = cust

                # Return to depot from cust
                b_dep = int((t - t_start) / bucket_size)
                if b_dep < 0:
                    b_dep = 0
                elif b_dep >= B:
                    b_dep = B - 1

                depot_arr = t + cost_data[b_dep, cust, 0]
                route_dur = depot_arr - t_start

                depot_late = 0.0
                if has_tw and depot_arr > depot_due_time:
                    depot_late = depot_arr - depot_due_time
                    if hard_time_windows:
                        continue

                duty_viol = 0.0
                if max_duty > 0.0 and route_dur > max_duty:
                    duty_viol = route_dur - max_duty
                    if hard_duty_limit:
                        continue

                arc_cost = route_dur + beta * (tw_viol + depot_late) + gamma * duty_viol
                cand = V[i] + arc_cost
                if cand < V[j]:
                    V[j] = cand

        return V[n]

    # Time-dependent constrained vehicle count with B > 1
    K = max_vehicles
    V_mat = np.full((n + 1, K + 1), 1e12, dtype=np.float64)
    V_mat[0, 0] = 0.0

    for v in range(1, K + 1):
        for i in range(n):
            if V_mat[i, v - 1] >= 1e12:
                continue
            load = 0.0
            t = t_start
            prev = 0
            tw_viol = 0.0

            for j in range(i + 1, n + 1):
                cust = tour[j - 1]
                load += demands[cust]
                if load > capacity:
                    break

                b = int((t - t_start) / bucket_size)
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1

                travel = cost_data[b, prev, cust]
                arr = t + travel

                if has_tw and ready_times is not None and due_times is not None:
                    if arr < ready_times[cust]:
                        start_serv = ready_times[cust]
                    else:
                        start_serv = arr
                        if arr > due_times[cust]:
                            lateness = arr - due_times[cust]
                            if hard_time_windows:
                                break
                            p_fac = priority_factors[cust] if has_priorities else 1.0
                            tw_viol += lateness * p_fac
                else:
                    start_serv = arr

                serv = service_times[cust] if has_services else 0.0
                t = start_serv + serv
                prev = cust

                b_dep = int((t - t_start) / bucket_size)
                if b_dep < 0:
                    b_dep = 0
                elif b_dep >= B:
                    b_dep = B - 1

                depot_arr = t + cost_data[b_dep, cust, 0]
                route_dur = depot_arr - t_start

                depot_late = 0.0
                if has_tw and depot_arr > depot_due_time:
                    depot_late = depot_arr - depot_due_time
                    if hard_time_windows:
                        continue

                duty_viol = 0.0
                if max_duty > 0.0 and route_dur > max_duty:
                    duty_viol = route_dur - max_duty
                    if hard_duty_limit:
                        continue

                arc_cost = route_dur + beta * (tw_viol + depot_late) + gamma * duty_viol
                cand = V_mat[i, v - 1] + arc_cost
                if cand < V_mat[j, v]:
                    V_mat[j, v] = cand

    min_cost = 1e12
    for v in range(1, K + 1):
        if V_mat[n, v] < min_cost:
            min_cost = V_mat[n, v]

    if min_cost < 1e12:
        return min_cost

    unc_cost = _split_cost_numba(
        tour, cost_data, demands, capacity, max_vehicles=-1,
        overflow_penalty=overflow_penalty, t_start=t_start,
        bucket_size=bucket_size, service_times=service_times,
        ready_times=ready_times, due_times=due_times,
        depot_due_time=depot_due_time, priority_factors=priority_factors,
        max_duty=max_duty, alpha=alpha, beta=beta, gamma=gamma,
        hard_time_windows=hard_time_windows, hard_duty_limit=hard_duty_limit,
    )
    return unc_cost + overflow_penalty


@njit
def _split_dag_with_pred(
    tour: np.ndarray,
    cost_data: np.ndarray,
    demands: np.ndarray,
    capacity: float,
    max_vehicles: int = -1,
    t_start: float = 0.0,
    bucket_size: float = 900.0,
    service_times: np.ndarray | None = None,
    ready_times: np.ndarray | None = None,
    due_times: np.ndarray | None = None,
    depot_due_time: float = 1e12,
    priority_factors: np.ndarray | None = None,
    max_duty: float = -1.0,
    alpha: float = 100.0,
    beta: float = 100.0,
    gamma: float = 100.0,
    hard_time_windows: bool = False,
    hard_duty_limit: bool = False,
) -> tuple[float, np.ndarray, int]:
    """Execute Prins Split returning shortest path cost, predecessor array, and best vehicle count."""
    n = len(tour)
    if n == 0:
        return 0.0, np.full(1, -1, dtype=np.int64), 0

    is_stationary = (cost_data.ndim == 2) or (cost_data.shape[0] == 1)
    has_services = service_times is not None
    has_priorities = priority_factors is not None

    has_tw = False
    if due_times is not None:
        for idx in range(len(due_times)):
            if due_times[idx] < 1e11:
                has_tw = True
                break

    # ------------------------------------------------------------------
    # 1. Stationary Path
    # ------------------------------------------------------------------
    if is_stationary:
        dist_matrix = cost_data if cost_data.ndim == 2 else cost_data[0]

        # Case A: Pure CVRP without time windows & duty limits
        if not has_tw and max_duty <= 0.0:
            if max_vehicles <= 0:
                V = np.full(n + 1, 1e12, dtype=np.float64)
                pred = np.full(n + 1, -1, dtype=np.int64)
                V[0] = 0.0

                for i in range(n):
                    if V[i] >= 1e12:
                        continue
                    load = 0.0
                    cost = 0.0

                    for j in range(i + 1, n + 1):
                        cust = tour[j - 1]
                        load += demands[cust]
                        if load > capacity:
                            break

                        if j == i + 1:
                            cost = dist_matrix[0, cust] + dist_matrix[cust, 0]
                        else:
                            prev = tour[j - 2]
                            cost = cost - dist_matrix[prev, 0] + dist_matrix[prev, cust] + dist_matrix[cust, 0]

                        cand = V[i] + cost
                        if cand < V[j]:
                            V[j] = cand
                            pred[j] = i

                return V[n], pred, 0

            # Constrained vehicle count
            K = max_vehicles
            V_mat = np.full((n + 1, K + 1), 1e12, dtype=np.float64)
            pred_mat = np.full((n + 1, K + 1), -1, dtype=np.int64)
            V_mat[0, 0] = 0.0

            for v in range(1, K + 1):
                for i in range(n):
                    if V_mat[i, v - 1] >= 1e12:
                        continue
                    load = 0.0
                    cost = 0.0

                    for j in range(i + 1, n + 1):
                        cust = tour[j - 1]
                        load += demands[cust]
                        if load > capacity:
                            break

                        if j == i + 1:
                            cost = dist_matrix[0, cust] + dist_matrix[cust, 0]
                        else:
                            prev = tour[j - 2]
                            cost = cost - dist_matrix[prev, 0] + dist_matrix[prev, cust] + dist_matrix[cust, 0]

                        cand = V_mat[i, v - 1] + cost
                        if cand < V_mat[j, v]:
                            V_mat[j, v] = cand
                            pred_mat[j, v] = i

            best_v = 1
            best_cost = 1e12
            for v in range(1, K + 1):
                if V_mat[n, v] < best_cost:
                    best_cost = V_mat[n, v]
                    best_v = v

            if best_cost < 1e12:
                pred_1d = np.full(n + 1, -1, dtype=np.int64)
                curr_node = n
                curr_layer = best_v
                while curr_node > 0 and curr_layer > 0:
                    p = int(pred_mat[curr_node, curr_layer])
                    pred_1d[curr_node] = p
                    curr_node = p
                    curr_layer -= 1
                return best_cost, pred_1d, best_v

            cost_unc, pred_unc, _ = _split_dag_with_pred(
                tour, cost_data, demands, capacity, max_vehicles=-1
            )
            return cost_unc, pred_unc, -1

        # Case B: Stationary CVRPTW
        if max_vehicles <= 0:
            V = np.full(n + 1, 1e12, dtype=np.float64)
            pred = np.full(n + 1, -1, dtype=np.int64)
            V[0] = 0.0

            for i in range(n):
                if V[i] >= 1e12:
                    continue
                load = 0.0
                t = t_start
                prev = 0
                route_dist = 0.0
                tw_viol = 0.0

                for j in range(i + 1, n + 1):
                    cust = tour[j - 1]
                    load += demands[cust]
                    if load > capacity:
                        break

                    travel = dist_matrix[prev, cust]
                    route_dist += travel
                    arr = t + travel

                    if has_tw and ready_times is not None and due_times is not None:
                        if arr < ready_times[cust]:
                            start_serv = ready_times[cust]
                        else:
                            start_serv = arr
                            if arr > due_times[cust]:
                                lateness = arr - due_times[cust]
                                if hard_time_windows:
                                    break
                                p_fac = priority_factors[cust] if has_priorities else 1.0
                                tw_viol += lateness * p_fac
                    else:
                        start_serv = arr

                    serv = service_times[cust] if has_services else 0.0
                    t = start_serv + serv
                    prev = cust

                    ret_travel = dist_matrix[cust, 0]
                    total_dist = route_dist + ret_travel
                    depot_arr = t + ret_travel

                    depot_late = 0.0
                    if has_tw and depot_arr > depot_due_time:
                        depot_late = depot_arr - depot_due_time
                        if hard_time_windows:
                            continue

                    route_dur = depot_arr - t_start
                    duty_viol = 0.0
                    if max_duty > 0.0 and route_dur > max_duty:
                        duty_viol = route_dur - max_duty
                        if hard_duty_limit:
                            continue

                    arc_cost = total_dist + beta * (tw_viol + depot_late) + gamma * duty_viol
                    cand = V[i] + arc_cost
                    if cand < V[j]:
                        V[j] = cand
                        pred[j] = i

            return V[n], pred, 0

        # Stationary CVRPTW with vehicle limit
        K = max_vehicles
        V_mat = np.full((n + 1, K + 1), 1e12, dtype=np.float64)
        pred_mat = np.full((n + 1, K + 1), -1, dtype=np.int64)
        V_mat[0, 0] = 0.0

        for v in range(1, K + 1):
            for i in range(n):
                if V_mat[i, v - 1] >= 1e12:
                    continue
                load = 0.0
                t = t_start
                prev = 0
                route_dist = 0.0
                tw_viol = 0.0

                for j in range(i + 1, n + 1):
                    cust = tour[j - 1]
                    load += demands[cust]
                    if load > capacity:
                        break

                    travel = dist_matrix[prev, cust]
                    route_dist += travel
                    arr = t + travel

                    if has_tw and ready_times is not None and due_times is not None:
                        if arr < ready_times[cust]:
                            start_serv = ready_times[cust]
                        else:
                            start_serv = arr
                            if arr > due_times[cust]:
                                lateness = arr - due_times[cust]
                                if hard_time_windows:
                                    break
                                p_fac = priority_factors[cust] if has_priorities else 1.0
                                tw_viol += lateness * p_fac
                    else:
                        start_serv = arr

                    serv = service_times[cust] if has_services else 0.0
                    t = start_serv + serv
                    prev = cust

                    ret_travel = dist_matrix[cust, 0]
                    total_dist = route_dist + ret_travel
                    depot_arr = t + ret_travel

                    depot_late = 0.0
                    if has_tw and depot_arr > depot_due_time:
                        depot_late = depot_arr - depot_due_time
                        if hard_time_windows:
                            continue

                    route_dur = depot_arr - t_start
                    duty_viol = 0.0
                    if max_duty > 0.0 and route_dur > max_duty:
                        duty_viol = route_dur - max_duty
                        if hard_duty_limit:
                            continue

                    arc_cost = total_dist + beta * (tw_viol + depot_late) + gamma * duty_viol
                    cand = V_mat[i, v - 1] + arc_cost
                    if cand < V_mat[j, v]:
                        V_mat[j, v] = cand
                        pred_mat[j, v] = i

        best_v = 1
        best_cost = 1e12
        for v in range(1, K + 1):
            if V_mat[n, v] < best_cost:
                best_cost = V_mat[n, v]
                best_v = v

        if best_cost < 1e12:
            pred_1d = np.full(n + 1, -1, dtype=np.int64)
            curr_node = n
            curr_layer = best_v
            while curr_node > 0 and curr_layer > 0:
                p = int(pred_mat[curr_node, curr_layer])
                pred_1d[curr_node] = p
                curr_node = p
                curr_layer -= 1
            return best_cost, pred_1d, best_v

        cost_unc, pred_unc, _ = _split_dag_with_pred(
            tour, cost_data, demands, capacity, max_vehicles=-1,
            t_start=t_start, bucket_size=bucket_size, service_times=service_times,
            ready_times=ready_times, due_times=due_times,
            depot_due_time=depot_due_time, priority_factors=priority_factors,
            max_duty=max_duty, alpha=alpha, beta=beta, gamma=gamma,
            hard_time_windows=hard_time_windows, hard_duty_limit=hard_duty_limit,
        )
        return cost_unc, pred_unc, -1

    # ------------------------------------------------------------------
    # 2. Time-Dependent Tensor Path
    # ------------------------------------------------------------------
    B = cost_data.shape[0]

    if max_vehicles <= 0:
        V = np.full(n + 1, 1e12, dtype=np.float64)
        pred = np.full(n + 1, -1, dtype=np.int64)
        V[0] = 0.0

        for i in range(n):
            if V[i] >= 1e12:
                continue
            load = 0.0
            t = t_start
            prev = 0
            tw_viol = 0.0

            for j in range(i + 1, n + 1):
                cust = tour[j - 1]
                load += demands[cust]
                if load > capacity:
                    break

                b = int((t - t_start) / bucket_size)
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1

                travel = cost_data[b, prev, cust]
                arr = t + travel

                if has_tw and ready_times is not None and due_times is not None:
                    if arr < ready_times[cust]:
                        start_serv = ready_times[cust]
                    else:
                        start_serv = arr
                        if arr > due_times[cust]:
                            lateness = arr - due_times[cust]
                            if hard_time_windows:
                                break
                            p_fac = priority_factors[cust] if has_priorities else 1.0
                            tw_viol += lateness * p_fac
                else:
                    start_serv = arr

                serv = service_times[cust] if has_services else 0.0
                t = start_serv + serv
                prev = cust

                b_dep = int((t - t_start) / bucket_size)
                if b_dep < 0:
                    b_dep = 0
                elif b_dep >= B:
                    b_dep = B - 1

                depot_arr = t + cost_data[b_dep, cust, 0]
                route_dur = depot_arr - t_start

                depot_late = 0.0
                if has_tw and depot_arr > depot_due_time:
                    depot_late = depot_arr - depot_due_time
                    if hard_time_windows:
                        continue

                duty_viol = 0.0
                if max_duty > 0.0 and route_dur > max_duty:
                    duty_viol = route_dur - max_duty
                    if hard_duty_limit:
                        continue

                arc_cost = route_dur + beta * (tw_viol + depot_late) + gamma * duty_viol
                cand = V[i] + arc_cost
                if cand < V[j]:
                    V[j] = cand
                    pred[j] = i

        return V[n], pred, 0

    # Constrained vehicle count with B > 1
    K = max_vehicles
    V_mat = np.full((n + 1, K + 1), 1e12, dtype=np.float64)
    pred_mat = np.full((n + 1, K + 1), -1, dtype=np.int64)
    V_mat[0, 0] = 0.0

    for v in range(1, K + 1):
        for i in range(n):
            if V_mat[i, v - 1] >= 1e12:
                continue
            load = 0.0
            t = t_start
            prev = 0
            tw_viol = 0.0

            for j in range(i + 1, n + 1):
                cust = tour[j - 1]
                load += demands[cust]
                if load > capacity:
                    break

                b = int((t - t_start) / bucket_size)
                if b < 0:
                    b = 0
                elif b >= B:
                    b = B - 1

                travel = cost_data[b, prev, cust]
                arr = t + travel

                if has_tw and ready_times is not None and due_times is not None:
                    if arr < ready_times[cust]:
                        start_serv = ready_times[cust]
                    else:
                        start_serv = arr
                        if arr > due_times[cust]:
                            lateness = arr - due_times[cust]
                            if hard_time_windows:
                                break
                            p_fac = priority_factors[cust] if has_priorities else 1.0
                            tw_viol += lateness * p_fac
                else:
                    start_serv = arr

                serv = service_times[cust] if has_services else 0.0
                t = start_serv + serv
                prev = cust

                b_dep = int((t - t_start) / bucket_size)
                if b_dep < 0:
                    b_dep = 0
                elif b_dep >= B:
                    b_dep = B - 1

                depot_arr = t + cost_data[b_dep, cust, 0]
                route_dur = depot_arr - t_start

                depot_late = 0.0
                if has_tw and depot_arr > depot_due_time:
                    depot_late = depot_arr - depot_due_time
                    if hard_time_windows:
                        continue

                duty_viol = 0.0
                if max_duty > 0.0 and route_dur > max_duty:
                    duty_viol = route_dur - max_duty
                    if hard_duty_limit:
                        continue

                arc_cost = route_dur + beta * (tw_viol + depot_late) + gamma * duty_viol
                cand = V_mat[i, v - 1] + arc_cost
                if cand < V_mat[j, v]:
                    V_mat[j, v] = cand
                    pred_mat[j, v] = i

    best_v = 1
    best_cost = 1e12
    for v in range(1, K + 1):
        if V_mat[n, v] < best_cost:
            best_cost = V_mat[n, v]
            best_v = v

    if best_cost < 1e12:
        pred_1d = np.full(n + 1, -1, dtype=np.int64)
        curr_node = n
        curr_layer = best_v
        while curr_node > 0 and curr_layer > 0:
            p = int(pred_mat[curr_node, curr_layer])
            pred_1d[curr_node] = p
            curr_node = p
            curr_layer -= 1
        return best_cost, pred_1d, best_v

    cost_unc, pred_unc, _ = _split_dag_with_pred(
        tour, cost_data, demands, capacity, max_vehicles=-1,
        t_start=t_start, bucket_size=bucket_size, service_times=service_times,
        ready_times=ready_times, due_times=due_times,
        depot_due_time=depot_due_time, priority_factors=priority_factors,
        max_duty=max_duty, alpha=alpha, beta=beta, gamma=gamma,
        hard_time_windows=hard_time_windows, hard_duty_limit=hard_duty_limit,
    )
    return cost_unc, pred_unc, -1


def prins_split(
    problem: Problem,
    tour: Sequence[int] | np.ndarray,
    max_vehicles: int | None = None,
    allow_extra_vehicles: bool = False,
    cost_oracle: Any | None = None,
    cost_tensor: CostTensor | None = None,
    max_duty_duration: float | None = None,
    hard_time_windows: bool = False,
    hard_duty_limit: bool = False,
    enforce_priority_early_service: bool = False,
    priority_early_ratio: float = 0.5,
    alpha: float = 100.0,
    beta: float = 100.0,
    gamma: float = 100.0,
) -> tuple[float, list[Route], bool, dict[str, list[str]]]:
    """Partition a giant tour into feasible vehicle routes adhering to capacity, time windows, and duty limits.

    Args:
        problem: Problem instance with depot, customers, and precomputed matrices.
        tour: Sequence of 1-based customer indices (e.g. 1 to n).
        max_vehicles: Optional limit on the number of vehicle routes.
        cost_oracle: Optional CostOracle for road network or traffic evaluation.
        cost_tensor: Optional precomputed CostTensor. If not provided, computed automatically.
        max_duty_duration: Maximum allowed driver duty duration per route.
        hard_time_windows: If True, time-window violations are strictly disallowed in Split.
        hard_duty_limit: If True, duty-limit violations are strictly disallowed in Split.
        enforce_priority_early_service: Tighten time window for priority customers.
        priority_early_ratio: Fraction of window for priority early service.
        alpha: Penalty weight for capacity violations.
        beta: Penalty weight for time window violations.
        gamma: Penalty weight for duty limit violations.

    Returns:
        (total_cost, routes, is_feasible, violations_dict)
    """
    tour_arr = np.asarray(tour, dtype=np.int64)
    n = len(tour_arr)
    if n == 0:
        return 0.0, [], True, {}

    if cost_tensor is None:
        cost_tensor = precompute_cost_tensor(problem, cost_oracle=cost_oracle)

    cost_data = cost_tensor.travel_time_tensor
    dist_mat = cost_tensor.distance_matrix

    num_nodes = problem.num_customers + 1
    demands = np.zeros(num_nodes, dtype=np.float64)
    services = np.zeros(num_nodes, dtype=np.float64)
    ready_times = np.zeros(num_nodes, dtype=np.float64)
    due_times = np.full(num_nodes, 1e12, dtype=np.float64)
    priority_factors = np.ones(num_nodes, dtype=np.float64)

    services[0] = problem.depot.service_time
    ready_times[0] = problem.depot.ready_time
    due_times[0] = problem.depot.due_time

    for idx, c in enumerate(problem.customers, start=1):
        demands[idx] = c.demand
        services[idx] = c.service_time
        ready_times[idx] = c.ready_time
        due_times[idx] = c.due_time
        priority_factors[idx] = 1.0 + (c.priority - 1) * 2.0

        if enforce_priority_early_service and c.priority > 1:
            early_due = c.ready_time + priority_early_ratio * (c.due_time - c.ready_time)
            due_times[idx] = early_due

    capacity = problem.vehicles[0].capacity if problem.vehicles else 100.0
    if max_vehicles is not None:
        limit_v = max_vehicles
    elif problem.vehicles and not allow_extra_vehicles:
        limit_v = len(problem.vehicles)
    else:
        limit_v = -1

    max_duty = -1.0
    if max_duty_duration is not None:
        max_duty = max_duty_duration
    elif problem.vehicles:
        v0 = problem.vehicles[0]
        if math.isfinite(v0.shift_end) and math.isfinite(v0.shift_start):
            diff = v0.shift_end - v0.shift_start
            if diff > 0:
                max_duty = diff

    cost, pred, best_v = _split_dag_with_pred(
        tour_arr,
        cost_data,
        demands,
        capacity,
        limit_v,
        t_start=cost_tensor.t_start,
        bucket_size=cost_tensor.bucket_size,
        service_times=services,
        ready_times=ready_times,
        due_times=due_times,
        depot_due_time=problem.depot.due_time,
        priority_factors=priority_factors,
        max_duty=max_duty,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        hard_time_windows=hard_time_windows,
        hard_duty_limit=hard_duty_limit,
    )

    # Backtrack route segments from pred array
    segments: list[list[int]] = []
    curr = n
    while curr > 0:
        p = int(pred[curr])
        if p < 0 or p >= curr:
            break
        seg = [int(tour_arr[k]) for k in range(p, curr)]
        segments.append(seg)
        curr = p

    segments.reverse()

    routes: list[Route] = []
    violations: dict[str, list[str]] = {}
    is_feasible = True

    if curr > 0:
        # Hard constraints pruned all valid DAG paths from 0 to n!
        # Fall back to soft/unconstrained split to reconstruct attempted routes and flag violations
        is_feasible = False
        violations.setdefault("feasibility", []).append(
            "Hard constraints violated; no feasible DAG path exists from depot"
        )
        cost_unc, pred_unc, _ = _split_dag_with_pred(
            tour_arr,
            cost_data,
            demands,
            capacity,
            -1,
            t_start=cost_tensor.t_start,
            bucket_size=cost_tensor.bucket_size,
            service_times=services,
            ready_times=ready_times,
            due_times=due_times,
            depot_due_time=problem.depot.due_time,
            priority_factors=priority_factors,
            max_duty=max_duty,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            hard_time_windows=False,
            hard_duty_limit=False,
        )
        curr = n
        segments = []
        while curr > 0:
            p = int(pred_unc[curr])
            if p < 0 or p >= curr:
                break
            segments.append([int(tour_arr[k]) for k in range(p, curr)])
            curr = p
        segments.reverse()

    eff_max_v = max_vehicles if max_vehicles is not None else (len(problem.vehicles) if (problem.vehicles and not allow_extra_vehicles) else None)
    if eff_max_v is not None and len(segments) > eff_max_v:
        is_feasible = False
        violations.setdefault("fleet_size", []).append(
            f"Required {len(segments)} routes, exceeding available fleet size of {eff_max_v}"
        )
        violations.setdefault("max_vehicles", []).append(
            f"Required {len(segments)} routes, exceeding max limit of {eff_max_v}"
        )

    # Build Route objects with time-dependent simulation
    vehicle_pool = problem.vehicles
    from routing.cost_oracle import Location

    stops_loc = [
        Location(id=problem.depot.id, lat=problem.depot.lat, lon=problem.depot.lon)
    ] + [Location(id=c.id, lat=c.lat, lon=c.lon) for c in problem.customers]

    total_dist = 0.0

    for r_idx, seg in enumerate(segments):
        if r_idx < len(vehicle_pool):
            veh_id = vehicle_pool[r_idx].id
        else:
            veh_id = f"V_{r_idx + 1}"
            if not allow_extra_vehicles:
                is_feasible = False
                violations.setdefault("fleet_size", []).append(
                    f"Route {r_idx + 1} assigned synthetic vehicle '{veh_id}' exceeding fleet size {len(vehicle_pool)}"
                )
        stop_ids: list[str] = []
        arrival_times: list[float] = []
        route_load = 0.0
        route_dist = 0.0

        current_time = problem.depot.ready_time
        prev_node = 0

        for cust_idx in seg:
            cust = problem.customers[cust_idx - 1]
            stop_ids.append(cust.id)
            route_load += cust.demand

            leg_dist = dist_mat[prev_node, cust_idx]
            if cost_oracle is not None:
                leg_time = cost_oracle.travel_time(
                    stops_loc[prev_node], stops_loc[cust_idx], depart_time=current_time
                )
            else:
                b = int((current_time - cost_tensor.t_start) / cost_tensor.bucket_size)
                b = max(0, min(cost_tensor.num_buckets - 1, b))
                leg_time = cost_data[b, prev_node, cust_idx]

            route_dist += leg_dist
            arrival = current_time + leg_time
            arrival_times.append(arrival)

            # Time window verification
            effective_due = cust.due_time
            if enforce_priority_early_service and cust.priority > 1:
                effective_due = cust.ready_time + priority_early_ratio * (cust.due_time - cust.ready_time)

            if arrival > effective_due:
                is_feasible = False
                violations.setdefault("time_window", []).append(
                    f"Stop {cust.id} reached at {arrival:.2f} > due time {effective_due:.2f}"
                )

            start_service = max(arrival, cust.ready_time)
            current_time = start_service + cust.service_time
            prev_node = cust_idx

        # Return to depot
        leg_dist = dist_mat[prev_node, 0]
        if cost_oracle is not None:
            leg_time = cost_oracle.travel_time(
                stops_loc[prev_node], stops_loc[0], depart_time=current_time
            )
        else:
            b = int((current_time - cost_tensor.t_start) / cost_tensor.bucket_size)
            b = max(0, min(cost_tensor.num_buckets - 1, b))
            leg_time = cost_data[b, prev_node, 0]

        route_dist += leg_dist
        final_depot_arrival = current_time + leg_time

        if final_depot_arrival > problem.depot.due_time:
            is_feasible = False
            violations.setdefault("depot_due_time", []).append(
                f"Vehicle {veh_id} returned at {final_depot_arrival:.2f} > {problem.depot.due_time:.2f}"
            )

        if route_load > capacity:
            is_feasible = False
            violations.setdefault("capacity", []).append(
                f"Vehicle {veh_id} load {route_load:.1f} > capacity {capacity:.1f}"
            )

        duration = final_depot_arrival - problem.depot.ready_time
        if max_duty > 0.0 and duration > max_duty:
            if hard_duty_limit:
                is_feasible = False
            violations.setdefault("driver_duty_limit", []).append(
                f"Vehicle {veh_id} duration {duration:.2f}s > max duty {max_duty:.2f}s"
            )

        total_dist += route_dist
        routes.append(
            Route(
                vehicle_id=veh_id,
                stop_ids=stop_ids,
                arrival_times=arrival_times,
                load=route_load,
                distance=route_dist,
                duration=duration,
            )
        )

    # For feasible solution: true distance. For infeasible: penalized cost
    solution_cost = total_dist if is_feasible else max(cost, total_dist + 10000.0)
    return solution_cost, routes, is_feasible, violations
