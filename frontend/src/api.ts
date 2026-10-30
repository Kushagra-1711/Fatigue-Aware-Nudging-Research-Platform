const API_BASE = import.meta.env.VITE_API_URL ?? "";

export interface Experiment {
  id: number;
  name: string;
  status: string;
  algorithm: string;
  fatigue_aware: boolean;
  mean_discounted_return: number | null;
  ci_95_lower: number | null;
  ci_95_upper: number | null;
  mean_health_score: number | null;
  mean_response_rate: number | null;
  mean_habituation: number | null;
  mean_nudges: number | null;
  error_message?: string | null;
  created_at?: string | null;
}

export interface Step {
  day: number;
  state_before: Record<string, number>;
  state_after: Record<string, number>;
  action: number;
  action_name: string;
  response_probability: number;
  responded: boolean;
  reward: number;
  habituation_before: number;
  habituation_after: number;
}

export interface PolicyEntry {
  state: number[];
  action: number;
  action_name: string;
  response_probability: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  dashboard: () => request<Record<string, unknown>>("/api/dashboard"),
  experiments: () =>
    request<{ experiments: Experiment[]; total: number }>("/api/experiments"),
  simulation: (body: unknown) =>
    request<{ steps: Step[]; discounted_return: number; algorithm: string }>(
      "/api/simulations",
      { method: "POST", body: JSON.stringify(body) },
    ),
  createExperiment: (body: unknown) =>
    request<Experiment>("/api/experiments", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  results: (id: number) =>
    request<{ evaluation: Record<string, unknown> | null; training_metrics: Record<string, unknown> | null }>(
      `/api/experiments/${id}/results`,
    ),
  policy: (id: number) =>
    request<{ entries: PolicyEntry[]; total_states: number }>(
      `/api/experiments/${id}/policy`,
    ),
  compare: (ids: number[]) =>
    request<{ experiments: Experiment[] }>("/api/compare", {
      method: "POST",
      body: JSON.stringify({ experiment_ids: ids }),
    }),
  export: (id: number, format: "json" | "csv") =>
    request<{ content: unknown; filename: string; format: string }>(
      `/api/experiments/${id}/export?format=${format}`,
    ),
};
