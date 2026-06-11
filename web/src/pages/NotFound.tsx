import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Music, Home } from "lucide-react";

export default function NotFound() {
  const navigate = useNavigate();

  return (
    <div className="max-w-md mx-auto px-4 py-20 text-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
      >
        <div className="text-7xl mb-6 animate-float">🎵</div>
        <h1 className="text-4xl font-display text-text mb-3">404</h1>
        <p className="text-text-light mb-2">这个页面不存在哦...</p>
        <p className="text-text-muted text-sm mb-8">
          也许你想找的是另一个平行世界里的 P 主？
        </p>

        <div className="flex items-center justify-center gap-3">
          <button onClick={() => navigate("/")} className="btn-primary">
            <Home size={16} />
            回到首页
          </button>
          <button onClick={() => navigate("/analyze")} className="btn-secondary">
            <Music size={16} />
            去分析
          </button>
        </div>
      </motion.div>
    </div>
  );
}
