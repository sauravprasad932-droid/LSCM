"""Unit tests for Google OR-Tools routing baseline in solver/ortools_baseline.py."""

from pathlib import Path
import pytest

from solver.io import load_cvrplib, load_solution_json
from solver.models import Customer, Depot, Problem, Vehicle
from solver.ortools_baseline import HAS_ORTOOLS, ORToolsConfig, solve_ortools
from solver.run import main

pytestmark = pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")


def test_ortools_hand_crafted() -> None:
    """Verify OR-Tools finds the optimal routes on the 4-customer toy instance."""
    depot = Depot(id="depot", lat=0.0, lon=0.0)
    customers = [
        Customer(id="1", lat=0.0, lon=1.0, demand=4.0, ready_time=0.0, due_time=100.0, service_time=0.0),
        Customer(id="2", lat=0.0, lon=2.0, demand=6.0, ready_time=0.0, due_time=100.0, service_time=0.0),
        Customer(id="3", lat=1.0, lon=0.0, demand=5.0, ready_time=0.0, due_time=100.0, service_time=0.0),
        Customer(id="4", lat=2.0, lon=0.0, demand=5.0, ready_time=0.0, due_time=100.0, service_time=0.0),
    ]
    vehicles = [Vehicle(id=f"V_{i+1}", capacity=10.0) for i in range(4)]
    problem = Problem(id="toy_ortools", depot=depot, customers=customers, vehicles=vehicles, coordinate_system="planar")
    problem.compute_euclidean_matrices()

    cfg = ORToolsConfig(time_limit_sec=2.0)
    solution = solve_ortools(problem, cfg)

    assert solution.is_feasible is True
    assert solution.violations == {}
    assert len(solution.routes) == 2
    assert pytest.approx(solution.total_cost, rel=1e-3) == 8.0

    all_visited = [cust_id for r in solution.routes for cust_id in r.stop_ids]
    assert set(all_visited) == {"1", "2", "3", "4"}


def test_ortools_on_cvrplib_instance() -> None:
    """Verify OR-Tools solves A-n32-k5 respecting capacities and customer coverage."""
    problem = load_cvrplib(Path("benchmarks/instances/A-n32-k5.vrp"))
    cfg = ORToolsConfig(time_limit_sec=5.0)

    solution = solve_ortools(problem, cfg)

    assert solution.is_feasible is True
    assert solution.violations == {}
    assert solution.algorithm == "OR-Tools"
    assert solution.total_cost > 0.0

    visited: list[str] = []
    capacity = problem.vehicles[0].capacity
    for route in solution.routes:
        assert route.load <= capacity + 1e-7, f"Route {route.vehicle_id} exceeded capacity"
        visited.extend(route.stop_ids)

    expected_ids = {c.id for c in problem.customers}
    assert set(visited) == expected_ids
    assert len(visited) == problem.num_customers


def test_ortools_cli_runner(tmp_path: Path) -> None:
    """Verify CLI runner with --solver ortools flag."""
    out_file = tmp_path / "ortools_sol.json"
    argv = [
        "--instance",
        "benchmarks/instances/A-n32-k5.vrp",
        "--solver",
        "ortools",
        "--time-limit",
        "3.0",
        "--out",
        str(out_file),
        "--quiet",
    ]

    ret = main(argv)
    assert ret == 0
    assert out_file.is_file()

    sol = load_solution_json(out_file)
    assert sol.algorithm == "OR-Tools"
    assert sol.is_feasible is True
    assert sol.total_cost > 0.0
