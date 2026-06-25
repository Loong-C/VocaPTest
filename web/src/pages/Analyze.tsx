import { useCallback, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, FileAudio, Info, RefreshCw, Sparkles } from "lucide-react";
import AudioUploader from "@/components/AudioUploader";
import LoadingSpinner from "@/components/LoadingSpinner";
import ScoreBar from "@/components/ScoreBar";
import { analyzeAudio } from "@/lib/api";
import { getProducerMeta } from "@/lib/producers";
import type { AnalyzeResult, SearchResultItem, UploadState } from "@/lib/types";

const RESULT_GRADIENTS = [
  "from-pink to-purple",
  "from-purple to-sky",
  "from-sky to-mint",
  "from-mint to-amber-400",
  "from-amber-400 to-pink",
];

export default function Analyze() {
  const [state, setState] = useState<UploadState>({ phase: "idle" });
  const [fileName, setFileName] = useState("");

  const handleFile = useCallback(async (file: File) => {
    setFileName(file.name);
    setState({ phase: "uploading", progress: 0 });

    try {
      const response = await analyzeAudio(file, (pct) => {
        if (pct >= 100) {
          setState({ phase: "analyzing" });
        } else {
          setState({ phase: "uploading", progress: pct });
        }
      });

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
    <div className="mx-auto max-w-2xl px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 text-center"
      >
        <h1 className="mb-2 font-display text-3xl text-text">
          <Sparkles className="mr-1 inline h-6 w-6 text-pink" />
          曲风分析
        </h1>
        <p className="text-sm text-text-light">上传一段音乐，发现你的风格匹配</p>
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
            <p className="mt-4 text-center text-xs leading-relaxed text-text-muted">
              上传音频仅用于本次分析，分析完成后会删除临时文件，不会长期保存。
            </p>
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
            <FileAudio className="mx-auto mb-4 h-12 w-12 animate-pulse text-pink" />
            <p className="mb-3 font-medium text-text">
              正在上传 <span className="text-pink-dark">{fileName}</span>
            </p>
            <div className="mb-2 h-2 overflow-hidden rounded-full bg-pink/10">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-pink to-purple"
                initial={{ width: 0 }}
                animate={{ width: `${state.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <p className="text-xs text-text-muted">{state.progress}%</p>
          </motion.div>
        )}

        {state.phase === "analyzing" && (
          <motion.div
            key="analyzing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card p-10 text-center"
          >
            <LoadingSpinner size="lg" text="上传完成，正在分析音频特征..." />
          </motion.div>
        )}

        {state.phase === "done" && <ResultView result={state.result} onReset={reset} />}

        {state.phase === "error" && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card p-8 text-center"
          >
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-red-50">
              <AlertTriangle className="h-7 w-7 text-red-400" />
            </div>
            <h3 className="mb-1 font-semibold text-text">分析失败</h3>
            <p className="mb-5 text-sm text-text-light">{state.message}</p>
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

function ResultView({ result, onReset }: { result: AnalyzeResult; onReset: () => void }) {
  const lowConfidence = result.accepted === false;
  const extraWarnings = result.warnings.filter(
    (warning) => !warning.includes("Low-confidence result")
  );

  return (
    <motion.div
      key="done"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-4"
    >
      {result.top_k.length > 0 && (
        <TopMatchCard
          item={result.top_k[0]!}
          accepted={!lowConfidence}
          confidence={result.confidence}
        />
      )}

      <div className="card space-y-5 p-6 stagger">
        <h3 className="text-center font-display text-lg text-text">
          {lowConfidence ? "灵感参考" : "匹配排名"}
        </h3>

        {result.top_k.length === 0 && (
          <p className="py-4 text-center text-sm text-text-muted">
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
              className="flex items-center gap-4 rounded-xl bg-white/50 p-3 transition-colors hover:bg-white/80"
            >
              <div
                className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-gradient-to-br ${meta.gradient} shadow-md`}
              >
                <span className="font-display text-sm text-white">
                  {item.display_name.slice(0, 2)}
                </span>
              </div>

              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-text">{item.display_name}</p>
                <div className="mt-0.5 flex gap-1">
                  {meta.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full bg-pink/5 px-1.5 py-0.5 text-[10px] text-text-muted"
                    >
                      {tag}
                    </span>
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

      {lowConfidence && (
        <div className="card flex items-start gap-3 border border-purple/10 bg-white/65 p-4 text-sm text-text-light">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-purple/10 text-purple">
            <Info size={16} />
          </div>
          <div>
            <p className="mb-1 font-medium text-text">你的音乐很有风格！</p>
            <p>
              它和当前资料库里的典型 P 主风格都不太像，仅列出最接近的候选作为参考。
            </p>
          </div>
        </div>
      )}

      {extraWarnings.length > 0 && (
        <div className="card flex items-start gap-3 bg-white/65 p-4 text-sm text-text-light">
          <Info size={16} className="mt-0.5 shrink-0 text-text-muted" />
          <div className="space-y-1">
            {extraWarnings.map((warning, index) => (
              <p key={index}>{warning}</p>
            ))}
          </div>
        </div>
      )}

      <div className="pt-2 text-center">
        <button onClick={onReset} className="btn-secondary">
          <RefreshCw size={16} />
          分析另一首歌
        </button>
      </div>
    </motion.div>
  );
}

function TopMatchCard({
  item,
  accepted,
  confidence,
}: {
  item: SearchResultItem;
  accepted: boolean;
  confidence: number | null;
}) {
  const meta = getProducerMeta(item.producer_slug);
  const pct = Math.round((confidence ?? item.score) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="card overflow-hidden"
    >
      <div
        className={`relative flex h-28 items-center justify-center bg-gradient-to-r ${
          accepted ? meta.gradient : "from-slate-400 to-purple-400"
        }`}
      >
        <div className="absolute inset-0 opacity-20">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute left-1/2 top-1/2 h-48 w-48 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white/40"
          />
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            className="absolute left-1/2 top-1/2 h-36 w-36 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/25"
          />
        </div>

        <div className="relative z-10 text-center">
          <p className="mb-1 text-sm text-white/80">
            {accepted ? "最匹配" : "最接近的灵感参考"}
          </p>
          <p className="font-display text-2xl text-white drop-shadow-lg">
            {item.display_name}
          </p>
        </div>
      </div>

      <div className="p-5 text-center">
        <p className="mb-3 text-sm text-text">
          {accepted ? "你的曲风听起来最像 " : "资料库里最接近的是 "}
          <span className="font-semibold text-pink-dark">{item.display_name}</span>
        </p>

        <div className="mb-4 flex flex-wrap justify-center gap-1.5">
          {meta.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full bg-pink/10 px-2.5 py-1 text-xs font-medium text-pink-dark"
            >
              {tag}
            </span>
          ))}
        </div>

        <div className="inline-flex items-baseline gap-1">
          <span className="font-display text-4xl text-text">{pct}</span>
          <span className="text-xl text-text-muted">%</span>
        </div>
        <p className="mt-1 text-xs text-text-muted">
          {accepted ? "模型置信度" : "参考置信度"}
        </p>
      </div>
    </motion.div>
  );
}
