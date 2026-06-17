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
    desc: "24kHz 单声道、20 秒窗口、10 秒步长、均匀覆盖且每首最多 12 段",
    color: "text-pink-dark",
    bg: "bg-pink/10",
  },
  {
    icon: Database,
    title: "参考库",
    desc: "31 位 P 主、376 首训练作品、62 首开发验证曲与 124 首最终冻结测试曲",
    color: "text-sky",
    bg: "bg-sky/10",
  },
  {
    icon: Code2,
    title: "后端",
    desc: "FastAPI + MERT 第 6 层歌曲均值 + 等先验 Shrinkage LDA + 校准拒识",
    color: "text-emerald-500",
    bg: "bg-mint/20",
  },
];

const PRODUCER_LIST = [
  "wowaka", "kemu", "Neru", "DECO*27", "ピノキオピー", "Mitchie M",
  "じん", "Orangestar", "cosMo@暴走P", "ハチ", "40mP", "ナユタン星人",
  "かいりきベア", "Kanaria", "Chinozo", "稲葉曇", "MIMI", "MARETU",
  "n-buna", "Ayase", "いよわ", "syudou", "なきそ", "すりぃ",
  "R Sound Design", "とあ", "てにをは", "煮ル果実", "はるまきごはん",
  "r-906", "sasakure.UK",
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
          用户上传一段音乐，系统在预先构建的 31 位 P 主参考库中寻找听感最接近的 Producer，
          输出 Top-K 相似结果。
        </p>
        <p className="text-text-light text-sm leading-relaxed">
          项目不追求严肃的"作者识别"，也不声称模型真正理解了作曲家的音乐学风格。
          当前模型把歌曲映射为 MERT 音频表征，再由歌曲级 Shrinkage LDA 给出候选结果。
          dev holdout 用于模型开发验证，final frozen 从不参与训练、模型选择或校准，
          用于监测扩类后的真实泛化表现。
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
          href="https://github.com/Loong-C/VocaPTest"
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
