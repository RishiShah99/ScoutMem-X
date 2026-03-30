import { useState } from 'react';

interface Props {
  log: string[];
}

export default function DecisionLog({ log }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="absolute top-4 left-4 z-10">
      <button
        onClick={() => setOpen(!open)}
        className="px-3 py-1.5 text-[10px] rounded-lg border border-white/10 bg-[#0d1117]/90 backdrop-blur-xl hover:bg-white/5 transition"
      >
        {open ? 'Hide Log' : 'Show Log'}
      </button>

      {open && (
        <div className="mt-2 w-80 max-h-72 overflow-y-auto rounded-xl border border-white/10 bg-[#0d1117]/90 backdrop-blur-xl p-3">
          <h3 className="text-xs font-bold text-amber-400 mb-2">DECISION LOG</h3>
          <div className="space-y-0.5 text-[10px] text-gray-400 leading-relaxed">
            {log.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
