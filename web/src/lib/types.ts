/** API response types — mirrors backend schemas.py */

export interface SearchResultItem {
  producer_slug: string;
  display_name: string;
  score: number;
  rank: number;
}

export interface AnalyzeResult {
  top_k: SearchResultItem[];
  accepted: boolean | null;
  confidence: number | null;
  margin: number | null;
  entropy: number | null;
  warnings: string[];
}

export interface AnalyzeResponse {
  job_id: string;
  status: string;
  result: AnalyzeResult | null;
  error: string | null;
}

export interface ProducerInfo {
  slug: string;
  display_name: string;
  song_count: number | null;
  segment_count: number | null;
  avatar_url: string | null;
  aliases: string[];
  profile_url: string | null;
  songs: ProducerSong[];
  training_songs: ProducerSong[];
  dev_song_count: number;
  dev_songs: ProducerSong[];
  frozen_song_count: number;
  frozen_songs: ProducerSong[];
  test_song_count: number;
  test_songs: ProducerSong[];
}

export interface ProducerSong {
  song_id: string;
  title: string;
  source_url: string | null;
}

export interface ProducerListResponse {
  producers: ProducerInfo[];
  backend: string | null;
  total_producers: number;
}

export interface HealthResponse {
  status: string;
  backend: string | null;
  producers_loaded: number;
}

/** UI-specific types */

export interface ProducerDisplay extends ProducerInfo {
  /** Gradient colors for avatar placeholder */
  gradient: string;
  /** Short description */
  style_tags: string[];
}

/** Upload state machine */
export type UploadState =
  | { phase: "idle" }
  | { phase: "uploading"; progress: number }
  | { phase: "analyzing" }
  | { phase: "done"; result: AnalyzeResult }
  | { phase: "error"; message: string };
