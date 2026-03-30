import type { ApiState } from '../api';

function confColor(c: number): string {
  if (c >= 0.8) return '#3fb950';
  if (c >= 0.5) return '#d29922';
  return '#f85149';
}

function stateBadge(state: string): string {
  switch (state) {
    case 'visible': return 'VERIFIED';
    case 'previously_seen': return 'SEEN';
    case 'uncertain': return 'UNC';
    case 'hypothesized': return 'WEAK';
    default: return state;
  }
}

interface Props {
  state: ApiState | null;
}

export default function MemoryPanel({ state }: Props) {
  if (!state) return null;

  return (
    <div className="absolute top-4 right-4 w-72 max-h-[calc(100vh-120px)] overflow-y-auto rounded-xl border border-white/10 bg-[#0d1117]/90 backdrop-blur-xl p-4 z-10">
      <h2 className="text-sm font-bold text-amber-400 mb-3 tracking-wide">SCENE MEMORY</h2>

      {/* Evidence gauge */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Evidence Sufficiency</span>
          <span>{state.evidence.toFixed(2)}</span>
        </div>
        <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${state.evidence * 100}%`,
              background: 'linear-gradient(90deg, #f85149, #d29922, #3fb950)',
            }}
          />
        </div>
      </div>

      {/* Object table — from real noisy perception */}
      <div className="space-y-1.5">
        {state.objects.map((obj, i) => (
          <div key={`${obj.label}-${obj.region}-${i}`} className="flex items-center gap-2 text-xs py-1 border-b border-white/5">
            <span className="flex-1 truncate" style={{ color: confColor(obj.confidence) }}>
              {obj.label}
            </span>
            <div className="w-12 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${obj.confidence * 100}%`,
                  backgroundColor: confColor(obj.confidence),
                }}
              />
            </div>
            <span className="text-gray-500 w-8 text-right">{obj.confidence.toFixed(2)}</span>
            <span className="text-[9px] px-1 rounded" style={{ color: confColor(obj.confidence) }}>
              {stateBadge(obj.state)}
            </span>
            <span className="text-gray-600 text-[9px] truncate w-14">{obj.region}</span>
          </div>
        ))}
      </div>

      {state.objects.length === 0 && (
        <p className="text-xs text-gray-600 mt-2">No objects detected yet. Click Explore.</p>
      )}

      {/* Stats */}
      <div className="mt-4 pt-3 border-t border-white/5 text-[10px] text-gray-500">
        <div>Step: {state.step} | Objects: {state.memoryCount}</div>
        <div>Waypoints: {state.waypointsVisited}/{state.waypointsTotal}</div>
        <div className="mt-1 text-gray-600">
          Confidence from noisy spatial perception
        </div>
      </div>
    </div>
  );
}
