interface Props {
  score: number;    // 0.0 – 1.0
  rank: number;
  colorClass?: string;
  className?: string;
}

export default function ScoreBar({ score, rank, colorClass = "from-pink to-purple", className = "" }: Props) {
  const pct = Math.round(score * 100);

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <span className="text-xs font-display text-text-muted w-6 text-right">
        #{rank}
      </span>
      <div className="flex-1 h-3 bg-pink/10 rounded-full overflow-hidden relative">
        <div
          className={`h-full bg-gradient-to-r ${colorClass} rounded-full
                      transition-all duration-1000 ease-out`}
          style={{ width: `${pct}%` }}
        />
        {/* Shimmer effect */}
        <div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent
                      animate-pulse"
          style={{ animationDuration: "2s" }}
        />
      </div>
      <span className="text-sm font-semibold text-text w-12 text-right tabular-nums">
        {pct}%
      </span>
    </div>
  );
}
