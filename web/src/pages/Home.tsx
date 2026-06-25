import { useNavigate } from "react-router-dom";
import { Sparkles, Music, ArrowRight } from "lucide-react";
import SparklesBg from "@/components/Sparkles";

const FLOATING_ELEMENTS = [
  { emoji: "🎵", className: "left-[8%] top-[15%] animate-float-slow", size: "text-3xl" },
  { emoji: "🎹", className: "left-[85%] top-[20%] animate-float", size: "text-2xl" },
  { emoji: "🌸", className: "left-[12%] top-[60%] animate-float", size: "text-xl" },
  { emoji: "⭐", className: "left-[78%] top-[55%] animate-float-slow", size: "text-2xl" },
  { emoji: "🎼", className: "left-[92%] top-[70%] animate-float", size: "text-xl" },
  { emoji: "💖", className: "left-[5%] top-[40%] animate-float", size: "text-lg" },
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="relative">
      {/* Floating decoration elements */}
      {FLOATING_ELEMENTS.map((el) => (
        <span
          key={el.emoji + el.className}
          className={`absolute select-none opacity-30 ${el.className} ${el.size}`}
        >
          {el.emoji}
        </span>
      ))}

      {/* Hero Section */}
      <section className="relative pt-16 pb-12 px-4 text-center">
        <SparklesBg />

        <div className="animate-scale-in relative z-10">
          <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                          bg-white/60 backdrop-blur-sm border border-pink-light/40
                          text-text-light text-sm mb-6">
            <Music size={14} className="text-pink" />
            Vocaloid Producer 风格匹配
          </span>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-display leading-tight mb-4">
            <span className="bg-gradient-to-r from-pink-dark via-purple to-sky bg-clip-text text-transparent">
              测测你的曲风
            </span>
            <br />
            <span className="text-text">最像哪位 P 主</span>
          </h1>

          <p className="max-w-lg mx-auto text-text-light text-base sm:text-lg leading-relaxed mb-8">
            上传一段音乐，AI 会在 31 位 Vocaloid Producer 的参考库中
            找到听感最接近的 P 主。<br />
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              onClick={() => navigate("/analyze")}
              className="btn-primary text-base group"
            >
              <Sparkles size={18} className="group-hover:animate-spin" />
              开始分析
              <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={() => navigate("/producers")}
              className="btn-secondary text-sm"
            >
              浏览 P 主列表
            </button>
          </div>
        </div>
      </section>

    </div>
  );
}
