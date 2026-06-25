import { useCallback, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, FileAudio, Info, RefreshCw, Sparkles } from "lucide-react";
import AudioUploader from "@/components/AudioUploader";
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

const PROCESS_STEPS = [
  "接收音频",
  "切分片段",
  "提取特征",
  "匹配风格",
];

const WAVEFORM_BARS = Array.from({ length: 22 }, (_, index) => index);
const ANALYZING_PROGRESS_WIDTHS = ["38%", "74%", "56%", "88%"];

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
          </motion.div>
        )}

        {state.phase === "uploading" && (
          <ProcessingCard
            key="uploading"
            phase="uploading"
            fileName={fileName}
            progress={state.progress}
          />
        )}

        {state.phase === "analyzing" && (
          <ProcessingCard
            key="analyzing"
            phase="analyzing"
            fileName={fileName}
          />
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

function ProcessingCard({
  phase,
  fileName,
  progress = 100,
}: {
  phase: "uploading" | "analyzing";
  fileName: string;
  progress?: number;
}) {
  const isUploading = phase === "uploading";
  const activeIndex = isUploading ? 0 : 2;
  const progressLabel = isUploading ? `${progress}%` : "分析中";
  const progressWidth = isUploading ? `${progress}%` : ANALYZING_PROGRESS_WIDTHS;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.98 }}
      className="card relative overflow-hidden p-8 text-center"
    >
      <div className="pointer-events-none absolute inset-x-8 top-6 h-24 rounded-full bg-gradient-to-r from-pink/15 via-purple/15 to-sky/15 blur-2xl" />

      <div className="relative mx-auto mb-5 flex h-28 max-w-sm items-end justify-center gap-1.5 rounded-3xl border border-white/60 bg-white/50 px-5 pb-5 shadow-inner">
        <motion.div
          animate={{ rotate: [0, -4, 4, 0], y: [0, -2, 0] }}
          transition={{ duration: 2.8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute left-5 top-5 flex h-10 w-10 items-center justify-center rounded-2xl bg-pink/10 text-pink-dark"
        >
          <FileAudio size={20} />
        </motion.div>

        {WAVEFORM_BARS.map((index) => (
          <motion.span
            key={index}
            animate={{
              height: [14, 34 + ((index * 7) % 28), 18 + ((index * 5) % 20), 14],
              opacity: [0.45, 0.95, 0.65, 0.45],
            }}
            transition={{
              duration: 1.15,
              repeat: Infinity,
              delay: index * 0.045,
              ease: "easeInOut",
            }}
            className="w-1.5 rounded-full bg-gradient-to-t from-pink via-purple to-sky"
          />
        ))}

        <motion.div
          animate={{ x: ["-120%", "120%"] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-0 left-1/2 h-px w-40 bg-gradient-to-r from-transparent via-white to-transparent"
        />
      </div>

      <p className="mb-1 font-medium text-text">
        {isUploading ? "正在上传" : "正在分析音频特征"}
        {fileName && (
          <span className="ml-1 text-pink-dark">{fileName}</span>
        )}
      </p>
      <p className="mb-5 text-xs text-text-muted">
        {isUploading
          ? "先把音频安全送达，再开始切片和提取 MERT 表征。"
          : "模型正在把片段均值、层融合概率和拒识校准合在一起。"}
      </p>

      <div className="mb-3 h-3 overflow-hidden rounded-full bg-pink/10">
        <motion.div
          className="relative h-full rounded-full bg-gradient-to-r from-pink via-purple to-sky"
          initial={{ width: isUploading ? 0 : "42%" }}
          animate={{ width: progressWidth }}
          transition={{
            duration: isUploading ? 0.3 : 1.8,
            repeat: isUploading ? 0 : Infinity,
            repeatType: "mirror",
            ease: "easeInOut",
          }}
        >
          <motion.div
            animate={{ x: ["-30%", "140%"] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
            className="absolute inset-y-0 w-1/2 bg-white/25 blur-sm"
          />
        </motion.div>
      </div>

      <div className="mb-5 flex items-center justify-between text-xs text-text-muted">
        <span>{progressLabel}</span>
        <span>{isUploading ? "上传阶段" : "特征分析阶段"}</span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {PROCESS_STEPS.map((step, index) => {
          const isActive = index === activeIndex || (!isUploading && index === 3);
          const isDone = index < activeIndex;
          return (
            <div
              key={step}
              className={`rounded-2xl px-3 py-2 text-xs transition-colors ${
                isActive
                  ? "bg-purple/10 text-purple"
                  : isDone
                  ? "bg-pink/10 text-pink-dark"
                  : "bg-white/55 text-text-muted"
              }`}
            >
              {step}
            </div>
          );
        })}
      </div>
    </motion.div>
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
          const tags = item.style_tags ?? [];
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
                  {tags.slice(0, 3).map((tag) => (
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
            <p className="mb-1 font-medium text-text">你的音乐很有特色！</p>
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
  const tags = item.style_tags ?? [];

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
          {tags.map((tag) => (
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
