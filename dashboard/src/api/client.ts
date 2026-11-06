import type {
  DynamicEventCreate,
  EventInjectResponse,
  HealthResponse,
  ProblemCreateRequest,
  ProblemResponse,
  SolutionResponse,
  SolveJob,
  SolveRequest,
} from './types';

const BASE = (import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '');

/** Format FastAPI/Pydantic error responses into readable strings.
 *  Handles 422 detail arrays: e.g. "body.customers.0.priority: Input should be >= 1"
 *  instead of "[object Object],[object Object]"
 */
function formatApiError(status: number, data: unknown): string {
  if (data && typeof data === 'object') {
    const d = data as Record<string, unknown>;
    if (Array.isArray(d.detail)) {
      // Pydantic v2 validation error array
      const msgs = (d.detail as Array<{ loc?: unknown[]; msg?: string }>).map((e) => {
        const loc = Array.isArray(e.loc) ? e.loc.join('.') : 'request';
        return `${loc}: ${e.msg ?? JSON.stringify(e)}`;
      });
      return `[${status}] Validation error:\n${msgs.join('\n')}`;
    }
    if (typeof d.detail === 'string') return `[${status}] ${d.detail}`;
    if (typeof d.message === 'string') return `[${status}] ${d.message}`;
    return `[${status}] ${JSON.stringify(data)}`;
  }
  return `[${status}] ${String(data)}`;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    let data: unknown = await res.text();
    try { data = JSON.parse(data as string); } catch {}
    throw new Error(formatApiError(res.status, data));
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: (): Promise<HealthResponse> => request('/health'),

  createProblem: (body: ProblemCreateRequest): Promise<ProblemResponse> =>
    request('/problems', { method: 'POST', body: JSON.stringify(body) }),

  listProblems: (): Promise<ProblemResponse[]> => request('/problems'),

  solveProblem: (problemId: string, body: SolveRequest): Promise<SolveJob> =>
    request(`/problems/${problemId}/solve`, { method: 'POST', body: JSON.stringify(body) }),

  getJob: (problemId: string, jobId: string): Promise<SolveJob> =>
    request(`/problems/${problemId}/jobs/${jobId}`),

  getSolution: (problemId: string): Promise<SolutionResponse> =>
    request(`/problems/${problemId}/solution`),

  injectEvent: (problemId: string, event: DynamicEventCreate): Promise<EventInjectResponse> =>
    request(`/problems/${problemId}/events`, { method: 'POST', body: JSON.stringify(event) }),

  deleteProblem: (problemId: string): Promise<void> =>
    request(`/problems/${problemId}`, { method: 'DELETE' }),
};
