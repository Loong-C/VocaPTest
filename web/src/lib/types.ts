/** API response types — mirrors backend schemas.py */

export interface SearchResultItem {
  producer_slug: string;
  display_name: string;
  score: number;
  rank: number;
}

export interface AnalyzeResult {
  top_k: SearchResultItem[];
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
