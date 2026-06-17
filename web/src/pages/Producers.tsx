import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ExternalLink, FlaskConical, LockKeyhole, Music, Search, Users, X } from "lucide-react";
import ProducerCard from "@/components/ProducerCard";
import LoadingSpinner from "@/components/LoadingSpinner";
import { getProducer, listProducers } from "@/lib/api";
import { enrichProducer } from "@/lib/producers";
import type { ProducerDisplay, ProducerInfo } from "@/lib/types";

export default function Producers() {
  const [producers, setProducers] = useState<ProducerDisplay[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<ProducerDisplay | null>(null);
  const [details, setDetails] = useState<ProducerInfo | null>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);

  useEffect(() => {
    listProducers()
      .then((data) => {
        setProducers(data.producers.map(enrichProducer));
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "加载失败");
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = search
    ? producers.filter(
        (producer) =>
          producer.display_name.toLowerCase().includes(search.toLowerCase()) ||
          producer.slug.toLowerCase().includes(search.toLowerCase()) ||
          producer.aliases.some((alias) =>
            alias.toLowerCase().includes(search.toLowerCase())
          ) ||
          producer.style_tags.some((tag) => tag.includes(search))
      )
    : producers;

  const openProducer = async (producer: ProducerDisplay) => {
    setSelected(producer);
    setDetails(null);
    setDetailsError(null);
    setDetailsLoading(true);
    try {
      setDetails(await getProducer(producer.slug));
    } catch (err) {
      setDetailsError(err instanceof Error ? err.message : "曲目目录加载失败");
    } finally {
      setDetailsLoading(false);
    }
  };

  const closeProducer = () => {
    setSelected(null);
    setDetails(null);
    setDetailsError(null);
  };

  useEffect(() => {
    if (!selected) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelected(null);
        setDetails(null);
        setDetailsError(null);
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selected]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="text-3xl font-display text-text mb-2">
          <Users className="inline w-6 h-6 text-purple mr-1" />
          Producer 图鉴
        </h1>
        <p className="text-text-light text-sm">
          {loading ? "加载中..." : `共收录 ${producers.length} 位 Vocaloid P 主`}
        </p>
      </motion.div>

      {!loading && !error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="max-w-md mx-auto mb-8"
        >
          <div className="relative">
            <Search
              size={18}
              className="absolute left-4 top-1/2 -translate-y-1/2 text-text-muted"
            />
            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="搜索 P 主名称、别名或风格标签..."
              className="w-full pl-11 pr-4 py-3 rounded-full
                         bg-white/60 border border-pink-light/30
                         text-text placeholder:text-text-muted
                         focus:outline-none focus:border-pink focus:ring-2
                         focus:ring-pink/20 transition-all text-sm"
            />
          </div>
        </motion.div>
      )}

      {loading && (
        <div className="py-20">
          <LoadingSpinner size="lg" text="加载 P 主数据..." />
        </div>
      )}

      {error && (
        <div className="card p-8 text-center">
          <p className="text-red-500 mb-3">{error}</p>
          <p className="text-text-muted text-sm">
            请确保后端 API 服务已启动 (localhost:8000)
          </p>
        </div>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="card p-12 text-center">
          <Music className="w-10 h-10 text-text-muted mx-auto mb-3" />
          <p className="text-text-light">没有找到匹配的 P 主</p>
          <p className="text-text-muted text-sm mt-1">试试其他关键词？</p>
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {filtered.map((producer, index) => (
            <motion.div
              key={producer.slug}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.04, duration: 0.4 }}
            >
              <ProducerCard
                producer={producer}
                onClick={() => openProducer(producer)}
              />
            </motion.div>
          ))}
        </div>
      )}

      <AnimatePresence>
        {selected && (
          <ProducerDialog
            producer={selected}
            details={details}
            loading={detailsLoading}
            error={detailsError}
            onClose={closeProducer}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

function ProducerDialog({
  producer,
  details,
  loading,
  error,
  onClose,
}: {
  producer: ProducerDisplay;
  details: ProducerInfo | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-slate-900/30 backdrop-blur-sm
                 flex items-center justify-center p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <motion.section
        role="dialog"
        aria-modal="true"
        aria-label={`${producer.display_name} 的训练、开发验证与最终冻结测试曲目`}
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 16, scale: 0.98 }}
        className="card w-full max-w-2xl max-h-[82vh] overflow-hidden"
      >
        <div className={`bg-gradient-to-r ${producer.gradient} p-6 text-white relative`}>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            className="absolute right-4 top-4 w-9 h-9 rounded-full bg-white/15
                       hover:bg-white/25 flex items-center justify-center transition-colors"
          >
            <X size={18} />
          </button>
          <div className="flex items-center gap-4 pr-12">
            <ProducerAvatar producer={producer} />
            <div>
              <h2 className="font-display text-2xl">{producer.display_name}</h2>
              {(details?.aliases ?? producer.aliases).length > 0 && (
                <p className="text-white/75 text-sm mt-1">
                  {(details?.aliases ?? producer.aliases).join(" · ")}
                </p>
              )}
              <p className="text-white/80 text-xs mt-2">
                {producer.song_count ?? 0} 首训练 ·{" "}
                {details?.dev_song_count ?? producer.dev_song_count} 首开发验证 ·{" "}
                {details?.frozen_song_count ?? producer.frozen_song_count ?? producer.test_song_count} 首最终冻结 ·{" "}
                {producer.segment_count ?? 0} 个训练片段
              </p>
            </div>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[58vh]">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <h3 className="font-display text-lg text-text">曲目分区</h3>
              <p className="text-xs text-text-muted mt-1">
                点击曲名可打开对应的公开视频来源；三类曲目在训练和评估中严格隔离
              </p>
            </div>
            {details?.profile_url && (
              <a
                href={details.profile_url}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-purple hover:text-pink-dark flex items-center gap-1"
              >
                VocaDB 资料
                <ExternalLink size={12} />
              </a>
            )}
          </div>

          {loading && <LoadingSpinner text="加载训练曲目..." />}

          {!loading && error && (
            <p className="text-red-500 text-sm py-8 text-center">{error}</p>
          )}

          {!loading && !error && details &&
            details.training_songs.length === 0 &&
            details.dev_songs.length === 0 &&
            details.frozen_songs.length === 0 && (
            <p className="text-text-muted text-sm py-8 text-center">
              暂未记录曲目
            </p>
          )}

          {!loading && !error && details && (
            <>
              <section>
                <div className="mb-4">
                  <h3 className="font-display text-lg text-text flex items-center gap-2">
                    <Music size={17} className="text-pink-dark" />
                    学习曲目
                  </h3>
                  <p className="text-xs text-text-muted mt-1">
                    实际参与当前模型训练的作品
                  </p>
                </div>
                <SongGrid songs={details.training_songs} tone="training" />
              </section>

              <div className="mt-7 pt-6 border-t border-purple/10">
                <div className="mb-4">
                  <h3 className="font-display text-lg text-text flex items-center gap-2">
                    <FlaskConical size={17} className="text-sky" />
                    开发验证曲目
                  </h3>
                  <p className="text-xs text-text-muted mt-1">
                    不参与训练，用于比较模型方案和观察错误
                  </p>
                </div>
                <SongGrid songs={details.dev_songs} tone="dev" />
              </div>

              <div className="mt-7 pt-6 border-t border-purple/10">
                <div className="mb-4">
                  <h3 className="font-display text-lg text-text flex items-center gap-2">
                    <LockKeyhole size={17} className="text-purple" />
                    最终冻结测试曲目
                  </h3>
                  <p className="text-xs text-text-muted mt-1">
                    仅用于最终验收，从未参与训练、模型选择或置信度校准
                  </p>
                </div>
                <SongGrid songs={details.frozen_songs} tone="frozen" />
              </div>
            </>
          )}
        </div>
      </motion.section>
    </motion.div>
  );
}

function SongGrid({
  songs,
  tone,
}: {
  songs: ProducerInfo["songs"];
  tone: "training" | "dev" | "frozen";
}) {
  if (songs.length === 0) {
    const label = tone === "training"
      ? "训练"
      : tone === "dev"
      ? "开发验证"
      : "最终冻结测试";
    return (
      <p className="text-text-muted text-sm py-4 text-center">
        暂未记录{label}曲目
      </p>
    );
  }
  return (
    <div className="grid sm:grid-cols-2 gap-2">
      {songs.map((song, index) => {
        const content = (
          <>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center
                              text-[11px] shrink-0 ${
              tone === "training"
                ? "bg-pink/10 text-pink-dark"
                : tone === "dev"
                ? "bg-sky/10 text-sky"
                : "bg-purple/10 text-purple"
            }`}>
              {index + 1}
            </span>
            <span className="min-w-0 flex-1 line-clamp-2">{song.title}</span>
            {song.source_url && (
              <ExternalLink size={13} className="shrink-0 opacity-50" />
            )}
          </>
        );
        const className =
          "flex items-center gap-2.5 rounded-xl hover:bg-white " +
          "border border-pink/5 px-3 py-2.5 text-sm text-text transition-colors " +
          (tone === "training"
            ? "bg-white/55"
            : tone === "dev"
            ? "bg-sky/[0.05]"
            : "bg-purple/[0.04]");
        return song.source_url ? (
          <a
            key={song.song_id}
            href={song.source_url}
            target="_blank"
            rel="noreferrer"
            className={className}
          >
            {content}
          </a>
        ) : (
          <div key={song.song_id} className={className}>
            {content}
          </div>
        );
      })}
    </div>
  );
}

function ProducerAvatar({ producer }: { producer: ProducerDisplay }) {
  const [imageAvailable, setImageAvailable] = useState(Boolean(producer.avatar_url));
  return (
    <div className="w-20 h-20 rounded-full bg-white/20 border-2 border-white/60
                    shadow-lg overflow-hidden flex items-center justify-center shrink-0">
      {producer.avatar_url && imageAvailable ? (
        <img
          src={producer.avatar_url}
          alt={producer.display_name}
          className="w-full h-full object-cover"
          onError={() => setImageAvailable(false)}
        />
      ) : (
        <span className="font-display text-xl">{producer.display_name.slice(0, 2)}</span>
      )}
    </div>
  );
}
