const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export async function getRuns() {
  return fetchJson("/api/runs");
}

export async function getRunDetail(runId) {
  return fetchJson(`/api/runs/${runId}`);
}

export async function getPredictions(runId, options = {}) {
  const params = new URLSearchParams();
  if (options.ticker) params.set("ticker", options.ticker);
  if (options.sortBy) params.set("sort_by", options.sortBy);
  if (typeof options.descending === "boolean") {
    params.set("descending", String(options.descending));
  }
  if (options.limit) params.set("limit", String(options.limit));
  const query = params.toString();
  return fetchJson(`/api/runs/${runId}/predictions${query ? `?${query}` : ""}`);
}
