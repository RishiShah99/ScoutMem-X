import { useEffect, useState } from 'react';

interface Props {
  message: string;
  isSuccess: boolean;
  details?: string;
  onDismiss: () => void;
}

export default function StatusToast({ message, isSuccess, details, onDismiss }: Props) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 8000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  const borderColor = isSuccess ? 'border-green-500/50' : 'border-red-500/50';
  const bgColor = isSuccess ? 'bg-green-500/10' : 'bg-red-500/10';
  const textColor = isSuccess ? 'text-green-400' : 'text-red-400';

  return (
    <div className={`absolute bottom-24 left-1/2 -translate-x-1/2 px-6 py-3 rounded-xl border ${borderColor} ${bgColor} backdrop-blur-xl z-20 text-center max-w-lg`}>
      <div className={`text-sm font-semibold ${textColor}`}>{message}</div>
      {details && <div className="text-xs text-gray-400 mt-1">{details}</div>}
    </div>
  );
}
