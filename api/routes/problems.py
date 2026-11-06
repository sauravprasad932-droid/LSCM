"""Problem management, async optimization jobs, and solution query endpoints."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from api.geojson import solution_to_geojson
from api.models import (
    ProblemCreateRequest,
    ProblemResponse,
    SolutionResponse,
    SolveJob,
    SolveRequest,
)
from api.store import ProblemStore, get_store
from api.websocket import manager
from routing.cost_oracle import (
    CostOracle,
    EuclideanOracle,
    SimulatedEventAdapter,
    TimeOfDayProfileAdapter,
    ValhallaOracle,
)
from solver.models import Customer, Depot, Problem, Solution, Vehicle
from solver.qpso import QPSOConfig, solve_qpso

router = APIRouter(prefix="/problems", tags=["Problems & Solver"])
logger = logging.getLogger(__name__)

# Dedicated thread pool for non-blocking execution of CPU-bound QPSO solver runs
_solver_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qpso_worker")


def _compute_bbox(depot: Depot, customers: list[Customer]) -> dict[str, float]:
    """Compute spatial bounding box across depot and customer stops."""
    lats = [depot.lat] + [c.lat for c in customers]
    lons = [depot.lon] + [c.lon for c in customers]
    return {
        "min_lat": min(lats),
        "min_lon": min(lons),
        "max_lat": max(lats),
        "max_lon": max(lons),
    }


def _run_solve_worker(
    problem: Problem,
    job: SolveJob,
    req: SolveRequest,
    store: ProblemStore,
) -> None:
    """Worker function executed in thread pool for asynchronous solving."""
    job.status = "running"
    store.save_job(job)
    t_start = time.perf_counter()

    # Broadcast job start
    manager.broadcast_sync(
        problem.id,
        {
            "type": "solve_started",
            "problem_id": problem.id,
            "job_id": job.job_id,
            "max_iters": req.max_iters,
            "num_particles": req.num_particles,
        },
    )

    try:
        # 1. Initialize CostOracle & TrafficAdapter
        traffic_adapter = SimulatedEventAdapter(
            base_adapter=TimeOfDayProfileAdapter({h: 1.0 for h in range(24)})
        )
        oracle: CostOracle
        if req.cost_oracle_type == "valhalla":
            oracle = ValhallaOracle(
                base_url=req.valhalla_url,
                traffic_adapter=traffic_adapter,
            )
        else:
            coord_sys = req.coordinate_system or problem.coordinate_system
            oracle = EuclideanOracle(
                base_speed_mps=8.333 if coord_sys == "geographic" else 1.0,
                coordinate_system=coord_sys,
                traffic_adapter=traffic_adapter,
            )
        store.save_oracle(problem.id, oracle)

        # 2. Setup Solver Configuration
        qpso_cfg = QPSOConfig(
            num_particles=req.num_particles,
            max_iters=req.max_iters,
            seed=req.seed,
            apply_tw_repair=req.apply_tw_repair,
            enforce_priority_early_service=req.enforce_priority_early_service,
        )

        # 3. Progress callback streaming to WebSocket subscribers
        def on_step(iter_num: int, best_cost: float) -> None:
            job.progress_iteration = iter_num
            job.current_cost = round(best_cost, 2)
            job.convergence_history.append({"iter": iter_num, "cost": round(best_cost, 2)})
            store.save_job(job)

            manager.broadcast_sync(
                problem.id,
                {
                    "type": "solve_progress",
                    "problem_id": problem.id,
                    "job_id": job.job_id,
                    "iteration": iter_num,
                    "max_iters": req.max_iters,
                    "gbest_cost": round(best_cost, 2),
                    "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
                },
            )

        # 4. Execute Optimization
        solution, _ = solve_qpso(
            problem,
            config=qpso_cfg,
            cost_oracle=oracle,
            callback=on_step,
        )

        # 5. Save Solution & Update Job
        job.status = "completed"
        job.completed_at = time.time()
        job.execution_time_sec = round(time.perf_counter() - t_start, 3)
        job.current_cost = round(solution.total_cost, 2)
        store.save_job(job)
        store.save_solution(problem.id, solution)

        # 6. Broadcast completion telemetry
        manager.broadcast_sync(
            problem.id,
            {
                "type": "solve_completed",
                "problem_id": problem.id,
                "job_id": job.job_id,
                "solution_cost": round(solution.total_cost, 2),
                "is_feasible": solution.is_feasible,
                "routes_count": len(solution.routes),
                "execution_time_sec": job.execution_time_sec,
            },
        )

    except Exception as exc:
        logger.exception(f"[SOLVER] Job {job.job_id} on problem {problem.id} failed: {exc}")
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = time.time()
        job.execution_time_sec = round(time.perf_counter() - t_start, 3)
        store.save_job(job)

        manager.broadcast_sync(
            problem.id,
            {
                "type": "solve_failed",
                "problem_id": problem.id,
                "job_id": job.job_id,
                "error": str(exc),
            },
        )


@router.post("", response_model=ProblemResponse, status_code=status.HTTP_201_CREATED, summary="Create Problem Instance")
def create_problem(
    req: ProblemCreateRequest,
    store: ProblemStore = Depends(get_store),
) -> ProblemResponse:
    """Register a new CVRPTW problem instance with depot and customer stops."""
    prob_id = req.id or f"prob_{uuid4().hex[:8]}"

    if store.get_problem(prob_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Problem with ID '{prob_id}' already exists",
        )

    # Supply default vehicle fleet if none provided
    vehicles = req.vehicles
    if not vehicles:
        default_capacity = sum(c.demand for c in req.customers) * 0.5
        vehicles = [
            Vehicle(
                id=f"V_{i+1}",
                capacity=max(50.0, default_capacity),
                shift_start=req.depot.ready_time,
                shift_end=req.depot.due_time,
            )
            for i in range(max(3, len(req.customers) // 3))
        ]

    problem = Problem(
        id=prob_id,
        depot=req.depot,
        customers=req.customers,
        vehicles=vehicles,
        coordinate_system=req.coordinate_system,
    )
    problem.compute_euclidean_matrices(speed=8.333 if req.coordinate_system == "geographic" else 1.0)
    store.save_problem(problem)

    bbox = _compute_bbox(req.depot, req.customers)
    return ProblemResponse(
        id=prob_id,
        depot=req.depot,
        num_customers=problem.num_customers,
        num_vehicles=problem.num_vehicles,
        bbox=bbox,
    )


@router.get("", response_model=list[ProblemResponse], summary="List Problems")
def list_problems(store: ProblemStore = Depends(get_store)) -> list[ProblemResponse]:
    """Retrieve metadata list of all stored problem instances."""
    results: list[ProblemResponse] = []
    for p in store.list_problems():
        bbox = _compute_bbox(p.depot, p.customers)
        results.append(
            ProblemResponse(
                id=p.id,
                depot=p.depot,
                num_customers=p.num_customers,
                num_vehicles=p.num_vehicles,
                bbox=bbox,
            )
        )
    return results


@router.get("/{problem_id}", response_model=Problem, summary="Get Problem Specification")
def get_problem(problem_id: str, store: ProblemStore = Depends(get_store)) -> Problem:
    """Retrieve full CVRPTW problem definition by ID."""
    problem = store.get_problem(problem_id)
    if problem is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem '{problem_id}' not found",
        )
    return problem


@router.delete("/{problem_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response, summary="Delete Problem")
def delete_problem(problem_id: str, store: ProblemStore = Depends(get_store)) -> Response:
    """Delete a problem instance and all its associated solutions and jobs."""
    deleted = store.delete_problem(problem_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem '{problem_id}' not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{problem_id}/solve", response_model=SolveJob, status_code=status.HTTP_202_ACCEPTED, summary="Launch Optimization Job")
def solve_problem(
    problem_id: str,
    req: SolveRequest = SolveRequest(),
    async_mode: bool = Query(
        False,
        description="Whether to run in background thread pool. Set True for async background worker; default False executes synchronously for serverless reliability.",
    ),
    store: ProblemStore = Depends(get_store),
) -> SolveJob:
    """Launch classical QPSO optimization and stream progress to WebSockets.

    In serverless platforms (e.g. Vercel), executing synchronously within the HTTP request
    guarantees the job completes without container suspension, returning full convergence history.
    """
    problem = store.get_problem(problem_id)
    if problem is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem '{problem_id}' not found",
        )

    job_id = f"job_{uuid4().hex[:8]}"
    job = SolveJob(
        job_id=job_id,
        problem_id=problem_id,
        status="pending",
        max_iters=req.max_iters,
    )
    store.save_job(job)

    if async_mode:
        # Launch non-blocking worker in solver thread pool
        _solver_executor.submit(_run_solve_worker, problem, job, req, store)
    else:
        # Execute synchronously to guarantee completion in serverless environments
        _run_solve_worker(problem, job, req, store)

    return job


@router.get("/{problem_id}/jobs/{job_id}", response_model=SolveJob, summary="Get Job Status")
def get_job_status(
    problem_id: str,
    job_id: str,
    store: ProblemStore = Depends(get_store),
) -> SolveJob:
    """Query current status and iteration progress of an active or finished solve job."""
    job = store.get_job(job_id)
    if job is None or job.problem_id != problem_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' for problem '{problem_id}' not found",
        )
    return job


@router.get("/{problem_id}/solution", response_model=SolutionResponse, summary="Get Active Fleet Solution")
def get_solution(
    problem_id: str,
    sim_time: float = Query(0.0, description="Simulation timestamp for visited stop flagging"),
    store: ProblemStore = Depends(get_store),
) -> SolutionResponse:
    """Retrieve active routing solution, visit ETAs, and GeoJSON features for MapLibre rendering."""
    problem = store.get_problem(problem_id)
    if problem is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem '{problem_id}' not found",
        )

    solution = store.get_solution(problem_id)
    if solution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No solution available for problem '{problem_id}'. Please trigger /solve first.",
        )

    oracle = store.get_oracle(problem_id)
    geojson_data = solution_to_geojson(problem, solution, cost_oracle=oracle, current_time=sim_time)

    return SolutionResponse(
        problem_id=problem_id,
        total_cost=round(solution.total_cost, 2),
        is_feasible=solution.is_feasible,
        violations=solution.violations,
        num_routes=len(solution.routes),
        routes=solution.routes,
        geojson=geojson_data,
        algorithm=solution.algorithm,
        execution_time_sec=round(solution.execution_time_sec, 3),
    )
