import { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '../api/client';
import { useWebSocket } from '../api/ws';
import type {
  SolutionResponse,
  SolveJob,
  WsMessage,
  ProblemCreateRequest,
  SolveRequest,
} from '../api/types';

export interface ConvergencePoint {
  iter: number;
  cost: number;
}

export interface SolverState {
  problemId: string | null;
  job: SolveJob | null;
  solution: SolutionResponse | null;
  previousSolution: SolutionResponse | null;  // before last event
  convergence: ConvergencePoint[];
  isLoading: boolean;
  error: string | null;
  oracleType: 'euclidean' | 'valhalla';
  valhallaConnected: boolean;
}

export interface SolverActions {
  loadScenario: (scenario: ProblemCreateRequest) => Promise<void>;
  solve: (params: Partial<SolveRequest>) => Promise<void>;
  fetchSolution: () => Promise<void>;
  setSolution: (solution: SolutionResponse) => void;
  reset: () => void;
}

const DEFAULT_STATE: SolverState = {
  problemId: null,
  job: null,
  solution: null,
  previousSolution: null,
  convergence: [],
  isLoading: false,
  error: null,
  oracleType: 'euclidean',
  valhallaConnected: false,
};

export function useSolver(): [SolverState, SolverActions] {
  const [state, setState] = useState<SolverState>(DEFAULT_STATE);
  const oracleRef = useRef<'euclidean' | 'valhalla'>('euclidean');

  const setError = (error: string | null) =>
    setState((s) => ({ ...s, error, isLoading: false }));

  const loadScenario = useCallback(async (scenario: ProblemCreateRequest) => {
    setState((s) => ({ ...s, isLoading: true, error: null }));
    try {
      // Delete existing problem if same ID exists
      if (scenario.id) {
        try { await api.deleteProblem(scenario.id); } catch {}
      }
      const prob = await api.createProblem(scenario);
      setState((s) => ({
        ...s,
        problemId: prob.id,
        job: null,
        solution: null,
        previousSolution: null,
        convergence: [],
        isLoading: false,
        error: null,
      }));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const solve = useCallback(
    async (params: Partial<SolveRequest>) => {
      setState((s) => {
        if (!s.problemId) return s;
        return { ...s, isLoading: true, error: null, convergence: [], job: null };
      });

      const problemId = state.problemId;
      if (!problemId) { setError('No problem loaded'); return; }

      const req: SolveRequest = {
        num_particles: params.num_particles ?? 25,
        max_iters: params.max_iters ?? 60,
        cost_oracle_type: params.cost_oracle_type ?? 'euclidean',
        valhalla_url: params.valhalla_url ?? 'http://localhost:8002',
        seed: params.seed ?? 42,
        apply_tw_repair: params.apply_tw_repair ?? true,
        enforce_priority_early_service: params.enforce_priority_early_service ?? false,
      };
      oracleRef.current = req.cost_oracle_type;

      try {
        const job = await api.solveProblem(problemId, req);

        // Immediate completion (synchronous / serverless execution)
        if (job.status === 'completed') {
          const conv = (job.convergence_history ?? []).map((pt) => ({
            iter: pt.iter,
            cost: pt.cost,
          }));
          const sol = await api.getSolution(problemId);
          setState((s) => ({
            ...s,
            job,
            solution: sol,
            convergence: conv.length > 0 ? conv : s.convergence,
            oracleType: req.cost_oracle_type,
            isLoading: false,
            error: null,
          }));
          return;
        }

        setState((s) => ({
          ...s,
          job,
          oracleType: req.cost_oracle_type,
          isLoading: true,
        }));
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [state.problemId],
  );

  const fetchSolution = useCallback(async () => {
    const pid = state.problemId;
    if (!pid) return;
    try {
      const sol = await api.getSolution(pid);
      setState((s) => ({ ...s, solution: sol, isLoading: false }));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [state.problemId]);

  const setSolution = useCallback((solution: SolutionResponse) => {
    setState((s) => ({ ...s, solution, isLoading: false }));
  }, []);

  const reset = useCallback(() => setState(DEFAULT_STATE), []);

  // Poll backend health to detect if Valhalla container is connected or offline
  useEffect(() => {
    let active = true;
    const pollHealth = () => {
      api.health()
        .then((h) => {
          if (active) {
            setState((s) => ({ ...s, valhallaConnected: h.valhalla_connected }));
          }
        })
        .catch(() => {
          if (active) {
            setState((s) => ({ ...s, valhallaConnected: false }));
          }
        });
    };

    pollHealth();
    const interval = setInterval(pollHealth, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  // WebSocket message handler
  const handleWsMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case 'solve_progress':
        setState((s) => ({
          ...s,
          convergence: [
            ...s.convergence,
            { iter: msg.iteration, cost: msg.gbest_cost },
          ],
          job: s.job
            ? { ...s.job, progress_iteration: msg.iteration, current_cost: msg.gbest_cost }
            : s.job,
        }));
        break;

      case 'solve_completed':
        setState((s) => ({ ...s, isLoading: false }));
        // Fetch solution after completion using problem_id from message or state
        {
          const pid = msg.problem_id || state.problemId;
          if (pid) {
            api.getSolution(pid)
              .then((sol) =>
                setState((ss) => ({ ...ss, solution: sol, isLoading: false })),
              )
              .catch(() => {});
          }
        }
        break;

      case 'solve_failed':
        setError(msg.error);
        break;

      case 'reoptimization_update':
        // Solution already fetched by useEvents; just update isLoading
        setState((s) => ({ ...s, isLoading: false }));
        break;

      default:
        break;
    }
  }, [state.problemId]); // eslint-disable-line

  useWebSocket(state.problemId, handleWsMessage);

  // Polling fallback when running in serverless or when WebSockets are unavailable
  useEffect(() => {
    if (!state.isLoading || !state.job || !state.problemId) return;
    if (state.job.status === 'completed' || state.job.status === 'failed') return;

    const pid = state.problemId;
    const jid = state.job.job_id;
    let active = true;

    const pollInterval = setInterval(async () => {
      try {
        const currentJob = await api.getJob(pid, jid);
        if (!active) return;

        if (currentJob.convergence_history && currentJob.convergence_history.length > 0) {
          setState((s) => ({
            ...s,
            convergence: currentJob.convergence_history!.map((pt) => ({
              iter: pt.iter,
              cost: pt.cost,
            })),
            job: currentJob,
          }));
        } else if (currentJob.progress_iteration > 0) {
          setState((s) => ({
            ...s,
            job: currentJob,
          }));
        }

        if (currentJob.status === 'completed') {
          clearInterval(pollInterval);
          const sol = await api.getSolution(pid);
          if (active) {
            setState((s) => ({ ...s, solution: sol, job: currentJob, isLoading: false }));
          }
        } else if (currentJob.status === 'failed') {
          clearInterval(pollInterval);
          if (active) {
            setState((s) => ({ ...s, error: currentJob.error_message || 'Optimization failed', isLoading: false }));
          }
        }
      } catch {
        // Continue polling until timeout or completion
      }
    }, 600);

    return () => {
      active = false;
      clearInterval(pollInterval);
    };
  }, [state.isLoading, state.job?.job_id, state.job?.status, state.problemId]);

  return [
    state,
    { loadScenario, solve, fetchSolution, setSolution, reset },
  ];
}
