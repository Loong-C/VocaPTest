import type { ProducerDisplay, ProducerSong } from "@/lib/types";

/**
 * Producer metadata: gradients, style tags, and display info.
 * These adorn the raw API data with visual presentation data.
 */
const PRODUCER_META: Record<string, { gradient: string; tags: string[] }> = {
  wowaka:         { gradient: "from-rose-400 to-pink-500",     tags: ["高速摇滚", "疾走感", "情感爆発"] },
  kemu:           { gradient: "from-red-500 to-orange-500",     tags: ["电子摇滚", "重低音", "青春叛逆"] },
  neru:           { gradient: "from-amber-500 to-yellow-500",   tags: ["摇滚", "电吉他", "中二病"] },
  deco27:         { gradient: "from-blue-400 to-cyan-500",      tags: ["流行", "恋爱", "catchy"] },
  pinocchiop:     { gradient: "from-lime-400 to-green-500",     tags: ["电波", "讽刺", "独特世界观"] },
  mitchie_m:      { gradient: "from-teal-400 to-emerald-500",   tags: ["调声拟真", "pop", "神调教"] },
  jin:            { gradient: "from-sky-400 to-indigo-500",     tags: ["钢琴摇滚", "物语性", "透明感"] },
  orangestar:     { gradient: "from-orange-400 to-amber-500",   tags: ["青春", "夏日", "吉他摇滚"] },
  cosmo:          { gradient: "from-violet-500 to-purple-600",  tags: ["爆速", "高BPM", "技术流"] },
  hachi:          { gradient: "from-gray-600 to-slate-700",     tags: ["和风", "电子", "独特世界"] },
  "40mp":         { gradient: "from-emerald-400 to-teal-500",   tags: ["清新", "日常", "温柔"] },
  nayutan:        { gradient: "from-fuchsia-400 to-pink-500",   tags: ["外星人", "电音", "洗脑"] },
  kairikibear:    { gradient: "from-purple-500 to-fuchsia-600", tags: ["暗黑", "狂气", "中毒性"] },
  kanaria:        { gradient: "from-rose-500 to-red-600",       tags: ["kawaii", "甜酷", "洗脑"] },
  chinozo:        { gradient: "from-cyan-400 to-blue-500",      tags: ["流行", "快节奏", "青春"] },
  inabakumori:    { gradient: "from-indigo-400 to-violet-500",  tags: ["和风", "抒情", "透明感"] },
  mimi:           { gradient: "from-green-400 to-lime-500",     tags: ["治愈", "轻快", "日常"] },
  maretu:         { gradient: "from-red-600 to-rose-700",       tags: ["暗黑", "重摇滚", "狂气"] },
  n_buna:         { gradient: "from-cyan-500 to-blue-600",      tags: ["清澈", "吉他", "叙情"] },
  ayase:          { gradient: "from-indigo-500 to-fuchsia-600", tags: ["都市感", "电子流行", "夜色"] },
  iyowa:          { gradient: "from-pink-400 to-orange-400",    tags: ["实验流行", "不安感", "复合节奏"] },
  syudou:         { gradient: "from-slate-600 to-red-600",      tags: ["暗黑流行", "锐利", "叙事感"] },
  nakiso:         { gradient: "from-zinc-700 to-purple-700",    tags: ["阴郁", "极简", "病态美"] },
  surii:          { gradient: "from-yellow-500 to-red-500",     tags: ["高速摇滚", "短篇冲击", "中毒性"] },
  r_sound_design: { gradient: "from-blue-500 to-violet-600",    tags: ["都市电子", "氛围感", "精致编曲"] },
  toa:            { gradient: "from-sky-300 to-pink-400",       tags: ["轻柔电子", "透明感", "细腻"] },
  teniwoha:       { gradient: "from-emerald-600 to-slate-700",  tags: ["和风", "文学性", "戏剧感"] },
};

export function getProducerMeta(slug: string): { gradient: string; tags: string[] } {
  return PRODUCER_META[slug] ?? { gradient: "from-pink-300 to-purple-400", tags: [] };
}

export function enrichProducer(producer: {
  slug: string;
  display_name: string;
  song_count: number | null;
  segment_count: number | null;
  avatar_url?: string | null;
  aliases?: string[];
  profile_url?: string | null;
  songs?: ProducerSong[];
  test_song_count?: number;
  test_songs?: ProducerSong[];
}): ProducerDisplay {
  const meta = getProducerMeta(producer.slug);
  return {
    ...producer,
    gradient: meta.gradient,
    style_tags: meta.tags,
    avatar_url: producer.avatar_url ?? null,
    aliases: producer.aliases ?? [],
    profile_url: producer.profile_url ?? null,
    songs: producer.songs ?? [],
    test_song_count: producer.test_song_count ?? 0,
    test_songs: producer.test_songs ?? [],
  };
}
