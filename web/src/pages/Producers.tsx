import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Users, Music, Search } from "lucide-react";
import ProducerCard from "@/components/ProducerCard";
import LoadingSpinner from "@/components/LoadingSpinner";
import { listProducers } from "@/lib/api";
import { enrichProducer } from "@/lib/producers";
import type { ProducerDisplay } from "@/lib/types";

export default function Producers() {
  const [producers, setProducers] = useState<ProducerDisplay[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

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
        (p) =>
          p.display_name.toLowerCase().includes(search.toLowerCase()) ||
          p.slug.toLowerCase().includes(search.toLowerCase()) ||
          p.style_tags.some((t) => t.includes(search))
      )
    : producers;

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      {/* Header */}
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

      {/* Search */}
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
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索 P 主名称或风格标签..."
              className="w-full pl-11 pr-4 py-3 rounded-full
                         bg-white/60 border border-pink-light/30
                         text-text placeholder:text-text-muted
                         focus:outline-none focus:border-pink focus:ring-2
                         focus:ring-pink/20 transition-all text-sm"
            />
          </div>
        </motion.div>
      )}

      {/* Content */}
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
          <p className="text-text-muted text-sm mt-1">
            试试其他关键词？
          </p>
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {filtered.map((p, i) => (
            <motion.div
              key={p.slug}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04, duration: 0.4 }}
            >
              <ProducerCard producer={p} />
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
