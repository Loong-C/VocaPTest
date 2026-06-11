import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, AlertTriangle, RefreshCw, FileAudio } from "lucide-react";
import AudioUploader from "@/components/AudioUploader";
import LoadingSpinner from "@/components/LoadingSpinner";
import ScoreBar from "@/components/ScoreBar";
import { analyzeAudio } from "@/lib/api";
import { getProducerMeta } from "@/lib/producers";
import type { UploadState, AnalyzeResult, SearchResultItem } from "@/lib/types";

const RESULT_GRADIENTS = [
  "from-pink to-purple",
  "from-purple to-sky",
  "from-sky to-mint",
  "from-mint to-amber-400",
  "from-amber-400 to-pink",
];

export default function Analyze() {
  const [state, setState] = useState<UploadState>({ phase: "idle" });
  const [fileName, setFileName] = useState<string>("");

  const handleFile = useCallback(async (file: File) => {
    setFileName(file.name);
    setState({ phase: "uploading", progress: 0 });

    try {
      const response = await analyzeAudio(file, (pct) => {
        setState({ phase: "uploading", progress: pct });
      });

      setState({ phase: "analyzing" });

      // Simulate a brief analysis phase for UX feel
      await new Promise((r) => setTimeout(r, 800));

      if (response.result) {
        setState({ phase: "done", result: response.result });
      } else if (response.error) {
        setState({ phase: "error", message: response.error });
      } else {
        setState({ phase: "error", message: "未知错误" });
      }
    } catch (err) {
      setState({
        phase: "error",
        message: err instanceof Error ? err.message : "上传失败",
      });
    }
  }, []);

  const reset = () => {
    setState({ phase: "idle" });
    setFileName("");
  };

  return (
    <div className="max-w-2xl mx-auto px-4 py-10">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="text-3xl font-display text-text mb-2">
          <Sparkles className="inline w-6 h-6 text-pink mr-1" />
          曲风分析
        </h1>
        <p className="text-text-light text-sm">上传一段音乐，发现你的风格匹配</p>
      </motion.div>

      <AnimatePresence mode="wait">
        {state.phase === "idle" && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <AudioUploader onFile={handleFile} />
          </motion.div>
        )}

        {state.phase === "uploading" && (
          <motion.div
            key="uploading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card p-8 text-center"
          >
            <FileAudio className="w-12 h-12 text-pink mx-auto mb-4 animate-pulse" />
            <p className="text-text font-medium mb-3">
              正在上传 <span className="text-pink-dark">{fileName}</span>
            </p>
            <div className="h-2 bg-pink/10 rounded-full overflow-hidden mb-2">
              <motion.div
                className="h-full bg-gradient-to-r from-pink to-purple rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${state.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <p className="text-text-muted text-xs">{state.progress}%</p>
          </motion.div>
        )}

        {state.phase === "analyzing" && (
          <motion.div
            key="analyzing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card p-12"
          >
            <LoadingSpinner size="lg" text="AI 正在分析音频特征..." />
          </motion.div>
        )}

        {state.phase === "done" && <ResultView result={state.result!} onReset={reset} />}

        {state.phase === "error" && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card p-8 text-center"
          >
            <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-red-50
                            flex items-center justify-center">
              <AlertTriangle className="w-7 h-7 text-red-400" />
            </div>
            <h3 className="text-text font-semibold mb-1">分析失败</h3>
            <p className="text-text-light text-sm mb-5">{state.message}</p>
            <button onClick={reset} className="btn-secondary">
              <RefreshCw size={16} />
              重新上传
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── Result View ── */
function ResultView({ result, onReset }: { result: AnalyzeResult; onReset: () => void }) {
  // Show warnings
  const hasWarnings = result.warnings.length > 0;

  return (
    <motion.div
      key="done"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-4"
    >
      {/* Top match highlight card */}
      {result.top_k.length > 0 && (
        <TopMatchCard item={result.top_k[0]!} />
      )}

      {/* All rankings */}
      <div className="card p-6 space-y-5 stagger">
        <h3 className="font-display text-lg text-text text-center">匹配排名</h3>

        {result.top_k.length === 0 && (
          <p className="text-text-muted text-sm text-center py-4">
            未找到匹配的 P 主，请尝试上传更长的音频片段
          </p>
        )}

        {result.top_k.map((item, i) => {
          const meta = getProducerMeta(item.producer_slug);
          return (
            <motion.div
              key={item.producer_slug}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className="flex items-center gap-4 p-3 rounded-xl
                         bg-white/50 hover:bg-white/80 transition-colors"
            >
              {/* Avatar */}
              <div
                className={`w-12 h-12 rounded-full bg-gradient-to-br ${meta.gradient}
                            flex items-center justify-center shadow-md shrink-0`}
              >
                <span className="text-white font-display text-sm">
                  {item.display_name.slice(0, 2)}
                </span>
              </div>

              <div className="flex-1 min-w-0">
                <p className="font-semibold text-text text-sm">{item.display_name}</p>
                <div className="flex gap-1 mt-0.5">
                  {meta.tags.slice(0, 3).map((t) => (
                    <span key={t} className="text-[10px] text-text-muted bg-pink/5
                                            px-1.5 py-0.5 rounded-full">{t}</span>
                  ))}
                </div>
              </div>

              <div className="w-32 shrink-0">
                <ScoreBar
                  score={item.score}
                  rank={item.rank}
                  colorClass={RESULT_GRADIENTS[i % RESULT_GRADIENTS.length]!}
                />
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Warnings */}
      {hasWarnings && (
        <div className="flex items-start gap-2 p-4 rounded-xl bg-amber-50 border border-amber-200
                        text-amber-700 text-sm">
          <AlertTriangle size={16} className="shrink-0 mt-0.5" />
          <div>
            {result.warnings.map((w, i) => (
              <p key={i}>{w}</p>
            ))}
          </div>
        </div>
      )}

      {/* Reset */}
      <div className="text-center pt-2">
        <button onClick={onReset} className="btn-secondary">
          <RefreshCw size={16} />
          分析另一首歌
        </button>
      </div>
    </motion.div>
  );
}

/* ── Top Match Hero Card ── */
function TopMatchCard({ item }: { item: SearchResultItem }) {
  const meta = getProducerMeta(item.producer_slug);
  const pct = Math.round(item.score * 100);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="card overflow-hidden"
    >
      <div className={`h-28 bg-gradient-to-r ${meta.gradient} relative
                       flex items-center justify-center`}>
        {/* Animated rings */}
        <div className="absolute inset-0 opacity-20">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-48 h-48 rounded-full border-2 border-white/40"
          />
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-36 h-36 rounded-full border border-white/25"
          />
        </div>

        <div className="relative z-10 text-center">
          <p className="text-white/80 text-sm mb-1">最匹配 🎯</p>
          <p className="text-white text-2xl font-display drop-shadow-lg">
            {item.display_name}
          </p>
        </div>
      </div>

      <div className="p-5 text-center">
        <p className="text-text text-sm mb-3">
          你的曲风听起来最像{" "}
          <span className="font-semibold text-pink-dark">{item.display_name}</span>
        </p>

        {/* Tags */}
        <div className="flex flex-wrap justify-center gap-1.5 mb-4">
          {meta.tags.map((tag) => (
            <span
              key={tag}
              className="px-2.5 py-1 text-xs rounded-full bg-pink/10 text-pink-dark font-medium"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Big score */}
        <div className="inline-flex items-baseline gap-1">
          <span className="text-4xl font-display text-text">{pct}</span>
          <span className="text-xl text-text-muted">%</span>
        </div>
        <p className="text-xs text-text-muted mt-1">相似度</p>
      </div>
    </motion.div>
  );
}
