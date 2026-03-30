const BASE = '';  // proxied via vite

export interface MemoryObject {
  label: string;
  confidence: number;
  state: string;
  frames: number;
  region: string;
  position: [number, number, number] | null;
}

export interface ApiState {
  objects: MemoryObject[];
  agent: [number, number, number];
  step: number;
  allExplored: boolean;
  memoryCount: number;
  evidence: number;
  waypointsTotal: number;
  waypointsVisited: number;
  log: string[];
}

export interface SearchResponse extends ApiState {
  found: boolean;
  confidence?: number;
  region?: string;
  targetPosition?: [number, number, number] | null;
  message?: string;
}

export interface InitWorldPayload {
  objects: { name: string; cleanName: string; position: [number, number, number] }[];
  bounds_min: [number, number, number];
  bounds_max: [number, number, number];
}

export async function initWorld(payload: InitWorldPayload): Promise<any> {
  const res = await fetch(`${BASE}/api/init-world`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function resetWorld(): Promise<ApiState> {
  const res = await fetch(`${BASE}/api/reset`, { method: 'POST' });
  return res.json();
}

export async function stepExplore(): Promise<ApiState & { done: boolean }> {
  const res = await fetch(`${BASE}/api/step`, { method: 'POST' });
  return res.json();
}

export async function autoExplore(): Promise<ApiState> {
  const res = await fetch(`${BASE}/api/explore`, { method: 'POST' });
  return res.json();
}

export async function searchObject(query: string): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/api/search?q=${encodeURIComponent(query)}`, { method: 'POST' });
  return res.json();
}

export async function getState(): Promise<ApiState> {
  const res = await fetch(`${BASE}/api/state`);
  return res.json();
}
