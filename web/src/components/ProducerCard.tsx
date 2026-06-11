import { Music } from "lucide-react";
import type { ProducerDisplay } from "@/lib/types";

interface Props {
  producer: ProducerDisplay;
  onClick?: () => void;
  className?: string;
}

export default function ProducerCard({ producer, onClick, className = "" }: Props) {
  const initials = producer.display_name.slice(0, 2);

  return (
    <button
      onClick={onClick}
      className={`card-hover w-full text-left group cursor-pointer
                  overflow-hidden ${className}`}
    >
      {/* Gradient header */}
      <div
        className={`h-20 bg-gradient-to-r ${producer.gradient} relative
                    flex items-center justify-center overflow-hidden`}
      >
        {/* Decorative rings */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                          w-32 h-32 rounded-full border-2 border-white/50" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                          w-24 h-24 rounded-full border border-white/30" />
        </div>

        {/* Avatar */}
        <div className="w-14 h-14 rounded-full bg-white/25 backdrop-blur-sm
                        flex items-center justify-center overflow-hidden
                        border-2 border-white/50 shadow-lg
                        group-hover:scale-110 transition-transform duration-300">
          {producer.avatar_url ? (
            <img
              src={producer.avatar_url}
              alt={producer.display_name}
              className="w-full h-full object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
          ) : null}
          <span className={`text-white font-display text-lg drop-shadow-md
                           ${producer.avatar_url ? "hidden" : ""}`}>
            {initials}
          </span>
        </div>
      </div>

      {/* Info */}
      <div className="p-4">
        <h3 className="font-semibold text-text text-base mb-1 group-hover:text-pink-dark
                       transition-colors">
          {producer.display_name}
        </h3>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {producer.style_tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-xs rounded-full
                         bg-pink/10 text-pink-dark font-medium"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Stats */}
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <span className="flex items-center gap-1">
            <Music size={12} />
            {producer.song_count ?? "?"} 首歌
          </span>
          <span>{producer.segment_count ?? "?"} 片段</span>
        </div>
      </div>
    </button>
  );
}
