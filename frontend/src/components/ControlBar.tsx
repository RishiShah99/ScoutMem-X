import { useState } from 'react';

interface Props {
  onReset: () => void;
  onStep: () => void;
  onExplore: () => void;
  onSearch: (query: string) => void;
  loading: boolean;
  allExplored: boolean;
}

export default function ControlBar({ onReset, onStep, onExplore, onSearch, loading, allExplored }: Props) {
  const [query, setQuery] = useState('find the red key');

  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 px-5 py-3 rounded-xl border border-white/10 bg-[#0d1117]/90 backdrop-blur-xl z-10">
      <button
        onClick={onReset}
        disabled={loading}
        className="px-4 py-2 text-xs rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition disabled:opacity-30"
      >
        Reset
      </button>
      <button
        onClick={onStep}
        disabled={loading || allExplored}
        className="px-4 py-2 text-xs rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition disabled:opacity-30"
      >
        Step
      </button>
      <button
        onClick={onExplore}
        disabled={loading || allExplored}
        className="px-5 py-2 text-xs rounded-lg bg-amber-600 hover:bg-amber-500 text-black font-semibold transition disabled:opacity-30"
      >
        Auto-Explore
      </button>

      <div className="w-px h-6 bg-white/10" />

      <input
        type="text"
        value={query}
        onChange={e => setQuery(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && onSearch(query)}
        placeholder="find the red key"
        className="w-48 px-3 py-2 text-xs rounded-lg bg-white/5 border border-white/10 outline-none focus:border-amber-500 transition"
      />
      <button
        onClick={() => onSearch(query)}
        disabled={loading}
        className="px-5 py-2 text-xs rounded-lg bg-amber-600 hover:bg-amber-500 text-black font-semibold transition disabled:opacity-30"
      >
        Search
      </button>
    </div>
  );
}
