"""Google OR-Tools routing baseline for CVRP and CVRPTW.

Serves as an exact and metaheuristic baseline for benchmarking against Route.it's QPSO solver.
Uses the identical distance matrix and demand specifications to guarantee rigorous comparability.
"""

from __future__ import annotations

import time
from typing import Sequence
import numpy as np
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except ImportError:
    pywrapcp = None
    routing_enums_pb2 = None
    HAS_ORTOOLS = False
from pydantic import BaseModel, Field

from solver.models import Problem, Route, Solution


class ORToolsConfig(BaseModel):
    """Configuration parameters for the OR-Tools baseline solver."""

    time_limit_sec: float = Field(10.0, gt=0.0, description="Solver runtime time limit in seconds")
    first_solution_strategy: str = Field(
        "PATH_CHEAPEST_ARC",
        description="Initial heuristic strategy (PATH_CHEAPEST_ARC, AUTOMATIC, etc.)",
    )
    metaheuristic: str = Field(
        "GUIDED_LOCAL_SEARCH",
        description="Local search metaheuristic (GUIDED_LOCAL_SEARCH, TABU_SEARCH, etc.)",
    )
    scale_factor: int = Field(
        10000,
        ge=1,
        description="Scaling factor to convert floating-point distances to integers for OR-Tools",
    )


def solve_ortools(
    problem: Problem,
    config: ORToolsConfig | None = None,
) -> Solution:
    """Solve the problem instance using Google OR-Tools Routing engine.

    Args:
        problem: Fully initialized Problem instance with precomputed matrices.
        config: ORToolsConfig options. If None, default settings are used.

    Returns:
        Solution model containing routes, costs, and feasibility metrics.
    """
    if not HAS_ORTOOLS:
        raise ImportError("Google OR-Tools is not installed. Install with 'pip install ortools'.")
    if config is None:
        config = ORToolsConfig()

    start_time = time.perf_counter()
    n = problem.num_customers
    num_vehicles = problem.num_vehicles

    if n == 0:
        return Solution(
            routes=[],
            total_cost=0.0,
            is_feasible=True,
            violations={},
            algorithm="OR-Tools",
            execution_time_sec=0.0,
            seed=None,
        )

    if problem.distance_matrix is None:
        problem.compute_euclidean_matrices()

    dist_mat = np.asarray(problem.distance_matrix, dtype=np.float64)
    scale = config.scale_factor

    # Scale distance matrix to integers for OR-Tools internal graph
    num_nodes = n + 1
    int_dist_mat = np.round(dist_mat * scale).astype(np.int64)

    # Depot is node 0
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # 1. Distance transit callback
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(int_dist_mat[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 2. Demand and Capacity dimension
    # Node 0 is depot (demand 0), nodes 1..n are customers
    demands = [0] + [int(round(c.demand * 100)) for c in problem.customers]
    vehicle_capacities = [int(round(v.capacity * 100)) for v in problem.vehicles]

    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,
        True,  # start cumul to zero
        "Capacity",
    )

    # 3. Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    # Initial heuristic
    strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy,
        config.first_solution_strategy,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    )
    search_parameters.first_solution_strategy = strategy

    # Metaheuristic
    meta = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic,
        config.metaheuristic,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    )
    search_parameters.local_search_metaheuristic = meta

    # Time limit
    sec = int(config.time_limit_sec)
    nanos = int((config.time_limit_sec - sec) * 1e9)
    search_parameters.time_limit.seconds = sec
    search_parameters.time_limit.nanos = nanos

    # 4. Solve
    solution_proto = routing.SolveWithParameters(search_parameters)
    elapsed_time = time.perf_counter() - start_time

    if not solution_proto:
        return Solution(
            routes=[],
            total_cost=float("inf"),
            is_feasible=False,
            violations={"ortools": ["No solution found within time limit"]},
            algorithm="OR-Tools",
            execution_time_sec=elapsed_time,
            seed=None,
        )

    # 5. Extract routes using our exact float distance matrix
    routes: list[Route] = []
    total_cost = 0.0

    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        stop_ids: list[str] = []
        route_load = 0.0
        route_dist = 0.0
        prev_node = 0

        # Traverse vehicle path
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                cust = problem.customers[node - 1]
                stop_ids.append(cust.id)
                route_load += cust.demand
                route_dist += dist_mat[prev_node, node]
                prev_node = node
            index = solution_proto.Value(routing.NextVar(index))

        # Return to depot if route served any customer
        if stop_ids:
            route_dist += dist_mat[prev_node, 0]
            veh_id = problem.vehicles[vehicle_idx].id
            routes.append(
                Route(
                    vehicle_id=veh_id,
                    stop_ids=stop_ids,
                    arrival_times=[0.0] * len(stop_ids),
                    load=route_load,
                    distance=route_dist,
                    duration=route_dist,
                )
            )
            total_cost += route_dist

    # Verify feasibility
    is_feasible = True
    violations: dict[str, list[str]] = {}

    visited_count = sum(len(r.stop_ids) for r in routes)
    if visited_count != n:
        is_feasible = False
        violations.setdefault("unassigned", []).append(
            f"Only {visited_count} of {n} customers served"
        )

    for r in routes:
        if r.load > problem.vehicles[0].capacity:
            is_feasible = False
            violations.setdefault("capacity", []).append(
                f"Vehicle {r.vehicle_id} load {r.load:.1f} exceeds capacity"
            )

    return Solution(
        routes=routes,
        total_cost=total_cost,
        is_feasible=is_feasible,
        violations=violations,
        algorithm="OR-Tools",
        execution_time_sec=elapsed_time,
        seed=None,
    )
