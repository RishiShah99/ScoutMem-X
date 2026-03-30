export interface RoomObject {
  label: string;
  confidence: number;
  state: 'visible' | 'previously_seen' | 'uncertain' | 'hypothesized';
  frames: number;
}

export interface Room {
  x: number;
  y: number;
  name: string;
  type: string;
  color: string;
  visited: boolean;
  objects: RoomObject[];
}

export interface AgentPos {
  x: number;
  y: number;
}

export interface AppState {
  rooms: Room[];
  agent: AgentPos;
  step: number;
  allExplored: boolean;
  memoryCount: number;
  evidence: number;
  log: string[];
}

export interface SearchResult extends AppState {
  found: boolean;
  path?: number[][];
  confidence?: number;
  room?: string;
  targetPos?: number[];
  message?: string;
}
