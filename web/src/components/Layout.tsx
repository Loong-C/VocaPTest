import { NavLink, Outlet } from "react-router-dom";
import { Github, Music, Users } from "lucide-react";
import FloatingNotes from "./FloatingNotes";

const REPOSITORY_URL = "https://github.com/Loong-C/VocaPTest";

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative min-h-screen flex flex-col">
      {/* Background floating decorations */}
      <FloatingNotes />

      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-cream/70 border-b border-pink-light/20">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <NavLink
            to="/"
            className="flex items-center gap-2 font-display text-lg text-text no-underline
                       hover:text-pink-dark transition-colors"
          >
            <span className="text-2xl">🎵</span>
            <span className="bg-gradient-to-r from-pink to-purple bg-clip-text text-transparent">
              VocaP Test
            </span>
          </NavLink>

          <nav className="flex items-center gap-1">
            <NavLink
              to="/analyze"
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3.5 py-2 rounded-full text-sm font-medium
                 transition-all duration-200 no-underline
                 ${isActive
                   ? "bg-pink/20 text-pink-dark"
                   : "text-text-light hover:text-text hover:bg-pink/10"}`
              }
            >
              <Music size={16} />
              <span className="hidden sm:inline">分析</span>
            </NavLink>

            <NavLink
              to="/producers"
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3.5 py-2 rounded-full text-sm font-medium
                 transition-all duration-200 no-underline
                 ${isActive
                   ? "bg-purple/20 text-purple-dark"
                   : "text-text-light hover:text-text hover:bg-purple/10"}`
              }
            >
              <Users size={16} />
              <span className="hidden sm:inline">P 主</span>
            </NavLink>

            <a
              href={REPOSITORY_URL}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub repository"
              title="GitHub"
              className="flex items-center justify-center w-9 h-9 rounded-full
                         text-text-light hover:text-text hover:bg-mint/20
                         transition-all duration-200 no-underline"
            >
              <Github size={17} />
            </a>
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="relative z-10 flex-1">
        {children}
        {/* Outlet for nested routes (not currently used, but future-proof) */}
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 text-center text-text-muted text-xs
                         border-t border-pink-light/15 bg-cream/80 backdrop-blur-sm">
        <p>
          仅供娱乐，不代表模型能真正识别作曲家风格
        </p>
        <p className="mt-1">
          Made with 💖 + FastAPI + MERT &nbsp;|&nbsp;
          <a
            href={REPOSITORY_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-pink-dark hover:underline"
          >
            GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}

// This component wraps the router outlet
export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  return <Layout>{children}</Layout>;
}
