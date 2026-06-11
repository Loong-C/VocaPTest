import { useEffect, useState } from "react";

interface Note {
  id: number;
  emoji: string;
  x: number;       // % from left
  delay: number;   // seconds
  duration: number; // seconds
  size: number;    // rem
  drift: number;   // horizontal drift px
}

const EMOJIS = ["🎵", "🎶", "🎼", "🌸", "⭐", "💖", "🎀", "✨", "🎹"];

function randomNote(id: number): Note {
  return {
    id,
    emoji: EMOJIS[Math.floor(Math.random() * EMOJIS.length)]!,
    x: Math.random() * 100,
    delay: Math.random() * 5,
    duration: 4 + Math.random() * 6,
    size: 1 + Math.random() * 1.5,
    drift: (Math.random() - 0.5) * 40,
  };
}

export default function FloatingNotes() {
  const [notes, setNotes] = useState<Note[]>([]);

  useEffect(() => {
    // Create initial batch
    const initial = Array.from({ length: 8 }, (_, i) => randomNote(i));
    setNotes(initial);

    // Periodically add new notes
    const timer = setInterval(() => {
      setNotes((prev) => {
        const next = [...prev.slice(-15), randomNote(Date.now())];
        return next;
      });
    }, 3000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0" aria-hidden>
      {notes.map((n) => (
        <span
          key={n.id}
          className="absolute bottom-0 opacity-25 select-none"
          style={{
            left: `${n.x}%`,
            fontSize: `${n.size}rem`,
            animation: `float-up ${n.duration}s ease-out ${n.delay}s forwards`,
            "--drift": `${n.drift}px`,
          } as React.CSSProperties}
        >
          {n.emoji}
        </span>
      ))}
    </div>
  );
}
