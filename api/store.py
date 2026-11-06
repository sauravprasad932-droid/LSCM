"""Storage layer interface and thread-safe in-memory implementation for Route.it API.

Enables full decoupling of API handlers from storage persistence.
Swappable to SQLite / persistent database without altering route logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from typing import TYPE_CHECKING

from api.models import SolveJob
from solver.models import Problem, Solution

if TYPE_CHECKING:
    from dynamic.reoptimizer import ReoptimizationRecord
    from routing.cost_oracle import CostOracle


class ProblemStore(ABC):
    """Abstract interface defining required storage operations for Route.it."""

    @abstractmethod
    def save_problem(self, problem: Problem) -> None:
        """Store or update a problem instance."""
        ...

    @abstractmethod
    def get_problem(self, problem_id: str) -> Problem | None:
        """Retrieve a problem instance by ID."""
        ...

    @abstractmethod
    def list_problems(self) -> list[Problem]:
        """List all stored problem instances."""
        ...

    @abstractmethod
    def delete_problem(self, problem_id: str) -> bool:
        """Delete a problem instance and its associated solutions/jobs."""
        ...

    @abstractmethod
    def save_solution(self, problem_id: str, solution: Solution) -> None:
        """Store or update the active fleet solution for a problem."""
        ...

    @abstractmethod
    def get_solution(self, problem_id: str) -> Solution | None:
        """Retrieve the active fleet solution for a problem."""
        ...

    @abstractmethod
    def save_job(self, job: SolveJob) -> None:
        """Store or update optimization job status."""
        ...

    @abstractmethod
    def get_job(self, job_id: str) -> SolveJob | None:
        """Retrieve job by its unique identifier."""
        ...

    @abstractmethod
    def add_event_record(self, problem_id: str, record: ReoptimizationRecord) -> None:
        """Append an audit record from a re-optimization event."""
        ...

    @abstractmethod
    def get_event_records(self, problem_id: str) -> list[ReoptimizationRecord]:
        """Retrieve the chronological audit history for a problem."""
        ...

    @abstractmethod
    def save_oracle(self, problem_id: str, oracle: CostOracle) -> None:
        """Cache the active cost oracle instance associated with a problem."""
        ...

    @abstractmethod
    def get_oracle(self, problem_id: str) -> CostOracle | None:
        """Retrieve the cached cost oracle instance associated with a problem."""
        ...


class InMemoryProblemStore(ProblemStore):
    """Thread-safe in-memory implementation of ProblemStore."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._problems: dict[str, Problem] = {}
        self._solutions: dict[str, Solution] = {}
        self._jobs: dict[str, SolveJob] = {}
        self._event_records: dict[str, list[ReoptimizationRecord]] = {}
        self._oracles: dict[str, CostOracle] = {}

    def save_problem(self, problem: Problem) -> None:
        with self._lock:
            self._problems[problem.id] = problem

    def get_problem(self, problem_id: str) -> Problem | None:
        with self._lock:
            prob = self._problems.get(problem_id)
            if prob is None and problem_id in ("vadodara", "vadodara_30"):
                try:
                    from benchmarks.demo_vadodara import build_vadodara_problem
                    prob = build_vadodara_problem()
                    self._problems[prob.id] = prob
                except Exception:
                    pass
            return prob

    def list_problems(self) -> list[Problem]:
        with self._lock:
            return list(self._problems.values())

    def delete_problem(self, problem_id: str) -> bool:
        with self._lock:
            if problem_id not in self._problems:
                return False
            del self._problems[problem_id]
            self._solutions.pop(problem_id, None)
            self._event_records.pop(problem_id, None)
            self._oracles.pop(problem_id, None)
            # Remove associated jobs
            job_ids_to_del = [j_id for j_id, job in self._jobs.items() if job.problem_id == problem_id]
            for j_id in job_ids_to_del:
                del self._jobs[j_id]
            return True

    def save_solution(self, problem_id: str, solution: Solution) -> None:
        with self._lock:
            self._solutions[problem_id] = solution

    def get_solution(self, problem_id: str) -> Solution | None:
        with self._lock:
            return self._solutions.get(problem_id)

    def save_job(self, job: SolveJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> SolveJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def add_event_record(self, problem_id: str, record: ReoptimizationRecord) -> None:
        with self._lock:
            self._event_records.setdefault(problem_id, []).append(record)

    def get_event_records(self, problem_id: str) -> list[ReoptimizationRecord]:
        with self._lock:
            return list(self._event_records.get(problem_id, []))

    def save_oracle(self, problem_id: str, oracle: CostOracle) -> None:
        with self._lock:
            self._oracles[problem_id] = oracle

    def get_oracle(self, problem_id: str) -> CostOracle | None:
        with self._lock:
            oracle = self._oracles.get(problem_id)
            if oracle is None:
                prob = self.get_problem(problem_id)
                if prob is not None:
                    try:
                        from routing.cost_oracle import EuclideanOracle, SimulatedEventAdapter, TimeOfDayProfileAdapter
                        adapter = SimulatedEventAdapter(
                            base_adapter=TimeOfDayProfileAdapter({h: 1.0 for h in range(24)})
                        )
                        oracle = EuclideanOracle(
                            base_speed_mps=8.333 if prob.coordinate_system == "geographic" else 1.0,
                            coordinate_system=prob.coordinate_system,
                            traffic_adapter=adapter,
                        )
                        self._oracles[problem_id] = oracle
                    except Exception:
                        pass
            return oracle

    def clear(self) -> None:
        """Clear all stored state (primarily for test resets)."""
        with self._lock:
            self._problems.clear()
            self._solutions.clear()
            self._jobs.clear()
            self._event_records.clear()
            self._oracles.clear()


class SQLiteProblemStore(ProblemStore):
    """Persistent SQLite-backed implementation of ProblemStore for serverless & local hosting."""

    def __init__(self, db_path: str | None = None) -> None:
        import os
        import sqlite3
        import tempfile

        if db_path is None:
            db_path = os.environ.get("SQLITE_PATH", os.path.join(tempfile.gettempdir(), "route_it.db"))
        self.db_path = db_path
        self._lock = threading.Lock()
        self._oracles: dict[str, CostOracle] = {}
        self._init_db()

    def _get_connection(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS problems (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL
                    );"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS solutions (
                        problem_id TEXT PRIMARY KEY,
                        data TEXT NOT NULL
                    );"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        problem_id TEXT NOT NULL,
                        data TEXT NOT NULL
                    );"""
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS event_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        problem_id TEXT NOT NULL,
                        data TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );"""
                )
                conn.commit()
            finally:
                conn.close()

    def save_problem(self, problem: Problem) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO problems (id, data) VALUES (?, ?);",
                    (problem.id, problem.model_dump_json()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_problem(self, problem_id: str) -> Problem | None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT data FROM problems WHERE id = ?;", (problem_id,))
                row = cursor.fetchone()
                if row is not None:
                    return Problem.model_validate_json(row[0])
            finally:
                conn.close()

        # Auto-seed canonical Vadodara problem if requested
        if problem_id in ("vadodara", "vadodara_30"):
            try:
                from benchmarks.demo_vadodara import build_vadodara_problem
                prob = build_vadodara_problem()
                self.save_problem(prob)
                return prob
            except Exception:
                pass
        return None

    def list_problems(self) -> list[Problem]:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT data FROM problems;")
                rows = cursor.fetchall()
            finally:
                conn.close()
        return [Problem.model_validate_json(r[0]) for r in rows]

    def delete_problem(self, problem_id: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("DELETE FROM problems WHERE id = ?;", (problem_id,))
                deleted = cursor.rowcount > 0
                if deleted:
                    conn.execute("DELETE FROM solutions WHERE problem_id = ?;", (problem_id,))
                    conn.execute("DELETE FROM jobs WHERE problem_id = ?;", (problem_id,))
                    conn.execute("DELETE FROM event_records WHERE problem_id = ?;", (problem_id,))
                    conn.commit()
                    self._oracles.pop(problem_id, None)
                return deleted
            finally:
                conn.close()

    def save_solution(self, problem_id: str, solution: Solution) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO solutions (problem_id, data) VALUES (?, ?);",
                    (problem_id, solution.model_dump_json()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_solution(self, problem_id: str) -> Solution | None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT data FROM solutions WHERE problem_id = ?;", (problem_id,))
                row = cursor.fetchone()
                if row is not None:
                    return Solution.model_validate_json(row[0])
                return None
            finally:
                conn.close()

    def save_job(self, job: SolveJob) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO jobs (job_id, problem_id, data) VALUES (?, ?, ?);",
                    (job.job_id, job.problem_id, job.model_dump_json()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_job(self, job_id: str) -> SolveJob | None:
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT data FROM jobs WHERE job_id = ?;", (job_id,))
                row = cursor.fetchone()
                if row is not None:
                    return SolveJob.model_validate_json(row[0])
                return None
            finally:
                conn.close()

    def add_event_record(self, problem_id: str, record: ReoptimizationRecord) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO event_records (problem_id, data, created_at) VALUES (?, ?, ?);",
                    (problem_id, record.model_dump_json(), record.timestamp),
                )
                conn.commit()
            finally:
                conn.close()

    def get_event_records(self, problem_id: str) -> list[ReoptimizationRecord]:
        from dynamic.reoptimizer import ReoptimizationRecord

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT data FROM event_records WHERE problem_id = ? ORDER BY id ASC;",
                    (problem_id,),
                )
                rows = cursor.fetchall()
            finally:
                conn.close()
        return [ReoptimizationRecord.model_validate_json(r[0]) for r in rows]

    def save_oracle(self, problem_id: str, oracle: CostOracle) -> None:
        with self._lock:
            self._oracles[problem_id] = oracle

    def get_oracle(self, problem_id: str) -> CostOracle | None:
        with self._lock:
            oracle = self._oracles.get(problem_id)
            if oracle is None:
                prob = self.get_problem(problem_id)
                if prob is not None:
                    try:
                        from routing.cost_oracle import EuclideanOracle, SimulatedEventAdapter, TimeOfDayProfileAdapter
                        adapter = SimulatedEventAdapter(
                            base_adapter=TimeOfDayProfileAdapter({h: 1.0 for h in range(24)})
                        )
                        oracle = EuclideanOracle(
                            base_speed_mps=8.333 if prob.coordinate_system == "geographic" else 1.0,
                            coordinate_system=prob.coordinate_system,
                            traffic_adapter=adapter,
                        )
                        self._oracles[problem_id] = oracle
                    except Exception:
                        pass
            return oracle

    def clear(self) -> None:
        """Clear all stored state."""
        with self._lock:
            self._oracles.clear()
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM problems;")
                conn.execute("DELETE FROM solutions;")
                conn.execute("DELETE FROM jobs;")
                conn.execute("DELETE FROM event_records;")
                conn.commit()
            finally:
                conn.close()


def _create_default_store() -> ProblemStore:
    """Create default ProblemStore: SQLite with fallback to InMemory."""
    try:
        return SQLiteProblemStore()
    except Exception:
        return InMemoryProblemStore()


# Global store instance for FastAPI dependency injection
_global_store: ProblemStore = _create_default_store()


def get_store() -> ProblemStore:
    """FastAPI dependency for accessing the active ProblemStore."""
    return _global_store


def set_store(store: ProblemStore) -> None:
    """Override active store (e.g. for testing or SQLite backend swap)."""
    global _global_store
    _global_store = store
