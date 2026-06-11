import { motion } from "framer-motion";
import { Sparkles, Code2, Database, Brain, Music, Github, Heart } from "lucide-react";

const TECH_ITEMS = [
  {
    icon: Brain,
    title: "AI 模型",
    desc: "MERT-v1-95M 预训练音乐理解模型，提取 768 维音频特征向量",
    color: "text-purple",
    bg: "bg-purple/10",
  },
  {
    icon: Music,
    title: "音频处理",
    desc: "30 秒无重叠切片、24kHz 单声道、RMS 能量过滤，确保特征质量",
    color: "text-pink-dark",
    bg: "bg-pink/10",
  },
  {
    icon: Database,
    title: "参考库",
    desc: "18 位 Vocaloid P 主 × 12 首代表作 × ~80 片段 = 1466 条嵌入向量",
    color: "text-sky",
    bg: "bg-sky/10",
  },
  {
    icon: Code2,
    title: "后端",
    desc: "FastAPI + Uvicorn，余弦相似度 + KMeans 多原型 Profile 检索",
    color: "text-emerald-500",
    bg: "bg-mint/20",
  },
];

const PRODUCER_LIST = [
  "wowaka", "kemu", "Neru", "DECO*27", "ピノキオピー", "Mitchie M",
  "じん", "Orangestar", "cosMo@暴走P", "ハチ", "40mP", "ナユタン星人",
  "かいりきベア", "Kanaria", "Chinozo", "稲葉曇", "MIMI", "MARETU",
];

export default function About() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-10"
      >
        <h1 className="text-3xl font-display text-text mb-2">
          <Sparkles className="inline w-6 h-6 text-pink mr-1" />
          关于项目
        </h1>
        <p className="text-text-light text-sm">了解 VocaP Test 的背后故事</p>
      </motion.div>

      {/* Project intro */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card p-6 mb-6"
      >
        <h2 className="font-display text-lg text-text mb-3">🎀 项目简介</h2>
        <p className="text-text-light text-sm leading-relaxed mb-3">
          VocaP Test 是一个娱乐向的 Vocaloid Producer 风格相似度系统。
          用户上传一段音乐，系统在预先构建的 18 位 P 主参考库中寻找听感最接近的 Producer，
          输出 Top-K 相似结果。
        </p>
        <p className="text-text-light text-sm leading-relaxed">
          项目不追求严肃的"作者识别"，也不声称模型真正理解了作曲家的音乐学风格。
          它更接近一个音频 Embedding 检索系统：把每位 P 主的代表作映射成向量空间中的风格原型，
          再把用户上传的歌曲映射到同一空间中计算相似度。
        </p>
      </motion.div>

      {/* Tech stack */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-6"
      >
        <h2 className="font-display text-lg text-text mb-4 text-center">
          🛠 技术栈
        </h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {TECH_ITEMS.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 + i * 0.08 }}
              className="card p-5 hover:shadow-[var(--shadow-kawaii-lg)] transition-shadow"
            >
              <div className={`w-10 h-10 rounded-xl ${item.bg} flex items-center justify-center mb-3`}>
                <item.icon className={`w-5 h-5 ${item.color}`} />
              </div>
              <h3 className="font-semibold text-text text-sm mb-1">{item.title}</h3>
              <p className="text-text-muted text-xs leading-relaxed">{item.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Producer coverage */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card p-6 mb-6"
      >
        <h2 className="font-display text-lg text-text mb-3">
          <Music className="inline w-5 h-5 text-pink mr-1" />
          覆盖的 P 主
        </h2>
        <div className="flex flex-wrap gap-2">
          {PRODUCER_LIST.map((name) => (
            <span
              key={name}
              className="px-3 py-1.5 text-sm rounded-full bg-pink/5 text-text
                         border border-pink-light/20 hover:border-pink/40
                         hover:bg-pink/10 transition-all"
            >
              {name}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Footer note */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="text-center space-y-2"
      >
        <p className="text-text-muted text-xs flex items-center justify-center gap-1">
          Made with <Heart className="w-3 h-3 text-pink fill-pink" /> + Python + React
        </p>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-text-muted text-xs
                     hover:text-pink-dark transition-colors no-underline"
        >
          <Github size={14} />
          GitHub
        </a>
      </motion.div>
    </div>
  );
}
