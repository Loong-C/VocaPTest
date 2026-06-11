import type {
  HealthResponse,
  ProducerListResponse,
  ProducerInfo,
  AnalyzeResponse,
} from "./types";

const BASE = ""; // Vite proxy handles /api → localhost:8000

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function listProducers(): Promise<ProducerListResponse> {
  return request<ProducerListResponse>("/api/producers");
}

export async function getProducer(slug: string): Promise<ProducerInfo> {
  return request<ProducerInfo>(`/api/producers/${slug}`);
}

export async function analyzeAudio(
  file: File,
  onProgress?: (pct: number) => void
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid response"));
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error")));

    xhr.open("POST", `${BASE}/api/analyze`);
    xhr.send(formData);
  });
}
