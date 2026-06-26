import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, FileAudio, Info, RefreshCw, Sparkles } from "lucide-react";
import AudioUploader from "@/components/AudioUploader";
import ScoreBar from "@/components/ScoreBar";
import { createAnalyzeJob, getAnalyzeJob } from "@/lib/api";
import { getProducerMeta } from "@/lib/producers";
import type { AnalyzeResult, JobStage, SearchResultItem, UploadState } from "@/lib/types";

const RESULT_GRADIENTS = [
  "from-pink to-purple",
  "from-purple to-sky",
  "from-sky to-mint",
  "from-mint to-amber-400",
  "from-amber-400 to-pink",
];

const PROCESS_STEPS: { stage: JobStage; label: string }[] = [
  { stage: "received", label: "接收音频" },
  { stage: "segmenting", label: "切分片段" },
  { stage: "embedding", label: "提取特征" },
  { stage: "classifying", label: "匹配风格" },
];

const WAVEFORM_BARS = Array.from({ length: 22 }, (_, index) => index);
const POLL_INTERVAL_MS = 900;
const STAGE_LABELS: Record<JobStage, string> = {
  received: "接收音频",
  segmenting: "切分片段",
  embedding: "提取特征",
  classifying: "匹配风格",
  done: "完成",
  failed: "失败",
};

const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export default function Analyze() {
  const [state, setState] = useState<UploadState>({ phase: "idle" });
  const [fileName, setFileName] = useState("");
  const runIdRef = useRef(0);

  const handleFile = useCallback(async (file: File) => {
    const runId = runIdRef.current + 1;
    runIdRef.current = runId;
    setFileName(file.name);
    setState({ phase: "uploading", progress: 0 });

    try {
      let job = await createAnalyzeJob(file, (pct) => {
        if (runIdRef.current === runId) {
          setState({ phase: "uploading", progress: pct });
        }
      });

      while (runIdRef.current === runId) {
        if (job.status === "done" && job.result) {
          setState({ phase: "done", result: job.result });
          return;
        }
        if (job.status === "failed") {
          setState({ phase: "error", message: job.error || "分析失败" });
          return;
        }
        if (job.status === "not_found") {
          setState({ phase: "error", message: job.error || "未找到分析任务" });
          return;
        }

        setState({
          phase: "analyzing",
          jobId: job.job_id,
          stage: job.stage,
          progress: job.progress,
        });
        await delay(POLL_INTERVAL_MS);
        job = await getAnalyzeJob(job.job_id);
      }
    } catch (err) {
      if (runIdRef.current === runId) {
        setState({
          phase: "error",
          message: err instanceof Error ? err.message : "上传失败",
        });
      }
    }
  }, []);

  const reset = () => {
    runIdRef.current += 1;
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
            progress={state.progress}
            stage={state.stage}
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
  stage = "received",
}: {
  phase: "uploading" | "analyzing";
  fileName: string;
  progress?: number;
  stage?: JobStage;
}) {
  const isUploading = phase === "uploading";
  const stageIndex = PROCESS_STEPS.findIndex((step) => step.stage === stage);
  const activeIndex = isUploading ? 0 : Math.max(0, stageIndex);
  const normalizedProgress = isUploading
    ? Math.min(Math.max(progress, 0), 100) / 100
    : Math.min(Math.max(progress, 0), 1);
  const progressPct = Math.round(normalizedProgress * 100);
  const progressLabel = isUploading ? `${progressPct}%` : `${progressPct}%`;
  const progressWidth = `${progressPct}%`;
  const stageLabel = isUploading ? "上传阶段" : STAGE_LABELS[stage];

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

      <div className="mb-3 h-3 overflow-hidden rounded-full bg-pink/10">
        <motion.div
          className="relative h-full rounded-full bg-gradient-to-r from-pink via-purple to-sky"
          initial={{ width: 0 }}
          animate={{ width: progressWidth }}
          transition={{
            duration: 0.35,
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
        <span>{stageLabel}</span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {PROCESS_STEPS.map((step, index) => {
          const isActive = index === activeIndex;
          const isDone = index < activeIndex;
          return (
            <div
              key={step.stage}
              className={`rounded-2xl px-3 py-2 text-xs transition-colors ${
                isActive
                  ? "bg-purple/10 text-purple"
                  : isDone
                  ? "bg-pink/10 text-pink-dark"
                  : "bg-white/55 text-text-muted"
              }`}
            >
              {step.label}
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
          {lowConfidence ? "相似参考" : "匹配排名"}
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
            {accepted ? "最匹配" : "最接近的相似参考"}
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
