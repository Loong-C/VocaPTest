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
            上传一段音乐，AI 会在 27 位 Vocaloid Producer 的参考库中
            找到听感最接近的 P 主。<br />
            <span className="text-text-muted text-sm">
              ✨ 娱乐向 · 非严肃作者识别
            </span>
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

      {/* Feature Cards */}
      <section className="max-w-4xl mx-auto px-4 pb-16 stagger">
        <div className="grid sm:grid-cols-3 gap-5">
          <div className="card p-6 text-center group hover:border-pink/40 transition-all duration-300">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-pink/10
                            flex items-center justify-center
                            group-hover:scale-110 transition-transform">
              <span className="text-2xl">🎧</span>
            </div>
            <h3 className="font-semibold text-text mb-1">上传音频</h3>
            <p className="text-text-muted text-sm">
              支持 WAV / MP3 / FLAC 等格式
            </p>
          </div>

          <div className="card p-6 text-center group hover:border-purple/40 transition-all duration-300">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-purple/10
                            flex items-center justify-center
                            group-hover:scale-110 transition-transform">
              <span className="text-2xl">🤖</span>
            </div>
            <h3 className="font-semibold text-text mb-1">AI 分析</h3>
            <p className="text-text-muted text-sm">
              MERT 音乐模型提取音频特征
            </p>
          </div>

          <div className="card p-6 text-center group hover:border-mint/60 transition-all duration-300">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-mint/20
                            flex items-center justify-center
                            group-hover:scale-110 transition-transform">
              <span className="text-2xl">🎯</span>
            </div>
            <h3 className="font-semibold text-text mb-1">风格匹配</h3>
            <p className="text-text-muted text-sm">
              返回 Top-5 最相似的 P 主
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}
