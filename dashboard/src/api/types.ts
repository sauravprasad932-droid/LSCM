// TypeScript mirrors of the FastAPI Pydantic models

export interface Depot {
  id: string;
  lat: number;
  lon: number;
  ready_time: number;
  due_time: number;
  service_time: number;
  name?: string;
}

export interface Customer {
  id: string;
  name?: string;
  lat: number;
  lon: number;
  demand: number;
  ready_time: number;
  due_time: number;
  service_time: number;
  // Backend: int ge=1. 1=standard, 2=high-priority. NOT a boolean.
  priority: number;
}

export interface Vehicle {
  id: string;
  capacity: number;
  shift_start: number;
  shift_end: number;
  cost_per_km?: number;
  cost_per_hour?: number;
}

// Matches solver.models.Route exactly.
export interface Route {
  vehicle_id: string;
  stop_ids: string[];        // ordered list of customer IDs visited
  arrival_times: number[];   // arrival time in seconds for each stop in stop_ids
  load: number;              // total demand on this route
  distance: number;
  duration: number;
  is_feasible?: boolean;     // may not be present in older responses
}

export interface ProblemCreateRequest {
  id?: string;
  depot: Depot;
  customers: Customer[];
  vehicles?: Vehicle[];
  coordinate_system?: 'geographic' | 'planar';
}

export interface ProblemResponse {
  id: string;
  depot: Depot;
  num_customers: number;
  num_vehicles: number;
  bbox: { min_lat: number; min_lon: number; max_lat: number; max_lon: number };
  created_at: number;
}

export interface SolveRequest {
  num_particles: number;
  max_iters: number;
  cost_oracle_type: 'euclidean' | 'valhalla';
  valhalla_url: string;
  coordinate_system?: 'geographic' | 'planar';
  seed: number;
  apply_tw_repair: boolean;
  enforce_priority_early_service: boolean;
}

export interface SolveJob {
  job_id: string;
  problem_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_iteration: number;
  max_iters: number;
  current_cost: number;
  convergence_history?: Array<{ iter: number; cost: number }>;
  error_message?: string;
  created_at: number;
  completed_at?: number;
  execution_time_sec: number;
}

export interface SolutionResponse {
  problem_id: string;
  total_cost: number;
  is_feasible: boolean;
  violations: Record<string, string[]>;
  num_routes: number;
  routes: Route[];
  geojson: GeoJsonFeatureCollection;
  algorithm: string;
  execution_time_sec: number;
}

export interface GeoJsonFeatureCollection {
  type: 'FeatureCollection';
  features: GeoJsonFeature[];
}

export interface GeoJsonFeature {
  type: 'Feature';
  geometry: {
    type: string;
    coordinates: number[] | number[][] | number[][][];
  };
  properties: Record<string, unknown>;
}

export interface DynamicEventCreate {
  event_type: 'edge_congestion' | 'road_blockage' | 'vehicle_delay';
  trigger_time: number;
  duration: number;
  origin_id?: string;
  dest_id?: string;
  multiplier?: number;
  description?: string;
}

export interface EventInjectResponse {
  event_id: string;
  tier_selected: string;
  wall_clock_time_ms: number;
  cost_before: number;
  cost_after: number;
  delta_f: number;
  stability_penalty: number;
  stops_changed_count: number;
  stops_changed: string[];
  is_feasible: boolean;
  message: string;
  solution: SolutionResponse;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
  valhalla_connected: boolean;
  active_problems_count: number;
  active_ws_connections: number;
}

// WebSocket message types
export type WsMessage =
  | { type: 'solve_started'; problem_id: string; job_id: string; max_iters: number; num_particles: number }
  | { type: 'solve_progress'; problem_id: string; job_id: string; iteration: number; max_iters: number; gbest_cost: number; elapsed_ms: number }
  | { type: 'solve_completed'; problem_id: string; job_id: string; solution_cost: number; is_feasible: boolean; routes_count: number; execution_time_sec: number }
  | { type: 'solve_failed'; problem_id: string; job_id: string; error: string }
  | { type: 'reoptimization_update'; problem_id: string; event_id: string; tier_selected: string; wall_clock_time_ms: number; cost_before: number; cost_after: number; delta_f: number; stops_changed_count: number; is_feasible: boolean; message: string };
