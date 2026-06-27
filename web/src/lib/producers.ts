import type { ProducerDisplay, ProducerSong } from "@/lib/types";

/**
 * Producer visual metadata.
 * Style tags come from the backend VocaDB cache, not from this frontend file.
 */
const PRODUCER_GRADIENTS: Record<string, string> = {
  wowaka: "from-rose-400 to-pink-500",
  kemu: "from-red-500 to-orange-500",
  neru: "from-amber-500 to-yellow-500",
  deco27: "from-blue-400 to-cyan-500",
  pinocchiop: "from-lime-400 to-green-500",
  mitchie_m: "from-teal-400 to-emerald-500",
  jin: "from-sky-400 to-indigo-500",
  orangestar: "from-orange-400 to-amber-500",
  cosmo: "from-violet-500 to-purple-600",
  hachi: "from-gray-600 to-slate-700",
  "40mp": "from-emerald-400 to-teal-500",
  nayutan: "from-fuchsia-400 to-pink-500",
  kairikibear: "from-purple-500 to-fuchsia-600",
  kanaria: "from-rose-500 to-red-600",
  chinozo: "from-cyan-400 to-blue-500",
  inabakumori: "from-indigo-400 to-violet-500",
  mimi: "from-green-400 to-lime-500",
  maretu: "from-red-600 to-rose-700",
  n_buna: "from-cyan-500 to-blue-600",
  ayase: "from-indigo-500 to-fuchsia-600",
  iyowa: "from-pink-400 to-orange-400",
  syudou: "from-slate-600 to-red-600",
  nakiso: "from-zinc-700 to-purple-700",
  surii: "from-yellow-500 to-red-500",
  r_sound_design: "from-blue-500 to-violet-600",
  toa: "from-sky-300 to-pink-400",
  teniwoha: "from-emerald-600 to-slate-700",
  niru_kajitsu: "from-stone-700 to-amber-600",
  harumaki_gohan: "from-indigo-300 to-sky-500",
  r_906: "from-cyan-600 to-teal-700",
  sasakure_uk: "from-violet-600 to-cyan-500",
  giga: "from-fuchsia-500 to-cyan-400",
  rerulili: "from-amber-400 to-rose-500",
  mikito_p: "from-lime-500 to-sky-500",
  hitoshizuku_p: "from-purple-500 to-red-500",
  balloon: "from-slate-600 to-violet-600",
  kuro_usa_p: "from-rose-600 to-stone-700",
  mothy: "from-yellow-600 to-red-700",
  hiiragi_magnetite: "from-cyan-500 to-fuchsia-500",
  owata_p: "from-orange-400 to-sky-500",
  nuyuri: "from-zinc-600 to-emerald-500",
  ryo: "from-sky-500 to-blue-700",
  eve: "from-red-500 to-zinc-700",
  papiyon: "from-violet-500 to-rose-500",
  wotaku: "from-slate-700 to-cyan-600",
  ume_tora: "from-fuchsia-500 to-amber-500",
  hachioji_p: "from-cyan-400 to-fuchsia-500",
  oster_project: "from-pink-400 to-amber-400",
  toma: "from-zinc-700 to-rose-600",
  gyari: "from-amber-400 to-cyan-500",
};

export function getProducerMeta(slug: string): { gradient: string } {
  return {
    gradient: PRODUCER_GRADIENTS[slug] ?? "from-pink-300 to-purple-400",
  };
}

export function withBasePath(url?: string | null): string | null {
  if (!url) return null;
  if (/^(https?:)?\/\//.test(url) || url.startsWith("data:") || url.startsWith("blob:")) {
    return url;
  }
  if (!url.startsWith("/")) return url;

  const baseUrl = import.meta.env.BASE_URL.endsWith("/")
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
  return `${baseUrl}${url.replace(/^\/+/, "")}`;
}

export function enrichProducer(producer: {
  slug: string;
  display_name: string;
  song_count: number | null;
  segment_count: number | null;
  avatar_url?: string | null;
  aliases?: string[];
  profile_url?: string | null;
  style_tags?: string[];
  style_tag_source?: string | null;
  style_tag_source_url?: string | null;
  songs?: ProducerSong[];
  training_songs?: ProducerSong[];
  dev_song_count?: number;
  dev_songs?: ProducerSong[];
  frozen_song_count?: number;
  frozen_songs?: ProducerSong[];
  test_song_count?: number;
  test_songs?: ProducerSong[];
}): ProducerDisplay {
  const meta = getProducerMeta(producer.slug);
  return {
    ...producer,
    gradient: meta.gradient,
    style_tags: producer.style_tags ?? [],
    style_tag_source: producer.style_tag_source ?? null,
    style_tag_source_url: producer.style_tag_source_url ?? null,
    avatar_url: withBasePath(producer.avatar_url),
    aliases: producer.aliases ?? [],
    profile_url: producer.profile_url ?? null,
    songs: producer.songs ?? [],
    training_songs: producer.training_songs ?? producer.songs ?? [],
    dev_song_count: producer.dev_song_count ?? 0,
    dev_songs: producer.dev_songs ?? [],
    frozen_song_count: producer.frozen_song_count ?? producer.test_song_count ?? 0,
    frozen_songs: producer.frozen_songs ?? producer.test_songs ?? [],
    test_song_count: producer.test_song_count ?? 0,
    test_songs: producer.test_songs ?? [],
  };
}
