"""Pydantic request and response schemas for the Route.it API.

Reuses domain models from solver.models and dynamic.events while providing
clean HTTP schemas for problem management, async solve jobs, GeoJSON payloads,
and dynamic traffic event injection.
"""

from __future__ import annotations

import time
from typing import Any, Literal
from pydantic import BaseModel, Field

from dynamic.events import EdgeCongestion, RoadBlockage, VehicleDelay
from solver.models import Customer, Depot, Problem, Route, Solution, Vehicle


class ProblemCreateRequest(BaseModel):
    """Payload for creating a new CVRPTW problem instance."""

    id: str | None = Field(None, description="Optional custom problem ID")
    depot: Depot = Field(..., description="Depot location, hours, and service turnaround")
    customers: list[Customer] = Field(..., min_length=1, description="List of customer delivery stops")
    vehicles: list[Vehicle] | None = Field(None, description="Optional vehicle fleet specifications")
    coordinate_system: Literal["geographic", "planar"] = Field(
        "geographic", description="Coordinate reference system ('geographic' or 'planar')"
    )


class ProblemResponse(BaseModel):
    """Metadata summary of a stored problem instance."""

    id: str
    depot: Depot
    num_customers: int
    num_vehicles: int
    bbox: dict[str, float] = Field(
        ...,
        description="Spatial bounding box {'min_lat': ..., 'min_lon': ..., 'max_lat': ..., 'max_lon': ...}",
    )
    created_at: float = Field(default_factory=time.time)


class SolveRequest(BaseModel):
    """Parameters for launching a classical QPSO solve job."""

    num_particles: int = Field(25, ge=5, le=200, description="Swarm size")
    max_iters: int = Field(60, ge=5, le=1000, description="Maximum iterations")
    cost_oracle_type: Literal["euclidean", "valhalla"] = Field(
        "euclidean", description="Routing oracle engine"
    )
    valhalla_url: str = Field("http://localhost:8002", description="Valhalla service URL")
    coordinate_system: Literal["geographic", "planar"] | None = Field(
        None, description="Coordinate system override ('geographic' or 'planar')"
    )
    seed: int = Field(42, description="Random seed for reproducibility")
    apply_tw_repair: bool = Field(True, description="Run time-window repair operator")
    enforce_priority_early_service: bool = Field(
        False, description="Enforce earlier service for priority customers"
    )


class SolveJob(BaseModel):
    """Internal and external state of an asynchronous optimization job."""

    job_id: str
    problem_id: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress_iteration: int = 0
    max_iters: int = 0
    current_cost: float = 0.0
    convergence_history: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None
    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    execution_time_sec: float = 0.0


class SolutionResponse(BaseModel):
    """Detailed routing solution with ETAs and MapLibre-ready GeoJSON features."""

    problem_id: str
    total_cost: float
    is_feasible: bool
    violations: dict[str, list[str]] = Field(default_factory=dict)
    num_routes: int
    routes: list[Route]
    geojson: dict[str, Any] = Field(
        ..., description="GeoJSON FeatureCollection containing Route LineStrings and Stop Points"
    )
    algorithm: str = "QPSO"
    execution_time_sec: float = 0.0


class DynamicEventCreate(BaseModel):
    """Payload for injecting a dynamic traffic incident into an active problem."""

    event_id: str | None = Field(None, description="Optional custom incident ID")
    event_type: Literal["edge_congestion", "road_blockage", "vehicle_delay"] = Field(
        ..., description="Type of dynamic event"
    )
    trigger_time: float = Field(0.0, ge=0.0, description="Simulation timestamp when event occurs")
    duration: float = Field(1800.0, gt=0.0, description="Incident duration in seconds")
    origin_id: str | None = Field(None, description="Corridor start stop ID")
    dest_id: str | None = Field(None, description="Corridor destination stop ID")
    edge_set: list[tuple[str, str]] | None = Field(
        None, description="Optional multi-edge blockage set"
    )
    multiplier: float = Field(2.0, ge=1.0, description="Slowdown multiplier for edge congestion")
    vehicle_id: str | None = Field(None, description="Target vehicle ID for vehicle delay")
    delay_seconds: float = Field(120.0, ge=0.0, description="Delay duration for vehicle delay")
    stop_id: str | None = Field(None, description="Stop location where vehicle delay occurs")
    description: str = Field("", description="Human-readable event description")

    def to_domain_event(self) -> EdgeCongestion | RoadBlockage | VehicleDelay:
        """Convert into a typed dynamic domain event."""
        evt_id = self.event_id or f"evt_{self.event_type}_{int(time.time() * 1000)}"

        if self.event_type == "edge_congestion":
            return EdgeCongestion(
                event_id=evt_id,
                trigger_time=self.trigger_time,
                duration=self.duration,
                origin_id=self.origin_id,
                dest_id=self.dest_id,
                edge_set=self.edge_set,
                multiplier=self.multiplier,
                description=self.description or f"Edge Congestion ({self.multiplier}x)",
            )
        elif self.event_type == "road_blockage":
            return RoadBlockage(
                event_id=evt_id,
                trigger_time=self.trigger_time,
                duration=self.duration,
                origin_id=self.origin_id,
                dest_id=self.dest_id,
                edge_set=self.edge_set,
                description=self.description or "Road Blockage",
            )
        elif self.event_type == "vehicle_delay":
            return VehicleDelay(
                event_id=evt_id,
                trigger_time=self.trigger_time,
                vehicle_id=self.vehicle_id or "V_1",
                delay_seconds=self.delay_seconds,
                stop_id=self.stop_id,
                description=self.description or f"Vehicle Delay ({self.delay_seconds}s)",
            )
        raise ValueError(f"Unsupported event_type: {self.event_type}")


class EventInjectResponse(BaseModel):
    """Outcome of tiered re-optimization triggered by an injected event."""

    event_id: str
    tier_selected: str
    wall_clock_time_ms: float
    cost_before: float
    cost_after: float
    delta_f: float
    stability_penalty: float
    stops_changed_count: int
    stops_changed: list[str] = Field(default_factory=list)
    is_feasible: bool
    message: str
    solution: SolutionResponse


class HealthResponse(BaseModel):
    """Operational health telemetry response."""

    status: str = "healthy"
    version: str
    uptime_seconds: float
    valhalla_connected: bool
    active_problems_count: int
    active_ws_connections: int
