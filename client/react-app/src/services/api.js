/**
 * All calls to the Banking Agentic AI service. Nothing in the UI holds customer data;
 * every number on screen comes back from one of these executions.
 */
const API_BASE = (process.env.REACT_APP_AGENTS_URL || '').replace(/\/$/, '');
let apiKey = process.env.REACT_APP_API_KEY || '';

export function setApiKey(key) { apiKey = key; }

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; this.auth = status === 401; }
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-API-Key': apiKey } : {}), ...(options.headers || {}) },
  });
  if (res.status === 401) throw new ApiError('API key required', 401);
  if (res.status === 429) throw new ApiError('Rate limit exceeded — retry shortly', 429);
  if (!res.ok) throw new ApiError(`${res.status} ${await res.text()}`, res.status);
  return res.json();
}

export const api = {
  health: () => request('/health'),
  customers: (limit = 40) => request(`/api/v2/customers?limit=${limit}`),
  /** Executes all six agents across the portfolio (template narratives only). */
  portfolio: (limit = 40, refresh = false) => request(`/api/v2/portfolio?limit=${limit}&refresh=${refresh ? 1 : 0}`, { method: 'POST' }),
  /** Executes the pipeline for one customer from the service's own data. */
  customer360: (id, refresh = false) => request(`/api/v2/customers/${id}/360?refresh=${refresh ? 1 : 0}`, { method: 'POST' }),
  /** Scores an in-flight transaction against the customer's baseline. */
  authorise: (customerId, transaction) =>
    request('/api/v2/agents/fraud/realtime', { method: 'POST', body: JSON.stringify({ customer_id: customerId, transaction, customer_data: {} }) }),
  chat: (customerId, message) => request('/api/v2/chat', { method: 'POST', body: JSON.stringify({ customer_id: customerId, message }) }),
};
