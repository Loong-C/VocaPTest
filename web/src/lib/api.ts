import type {
  HealthResponse,
  ProducerListResponse,
  ProducerInfo,
  AnalyzeResponse,
  JobStatusResponse,
} from "./types";

const APP_BASE = import.meta.env.BASE_URL === "/"
  ? ""
  : import.meta.env.BASE_URL.replace(/\/$/, "");

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
    signal: controller.signal,
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(formatApiError(res.status, body, res.statusText));
  }
  return await res.json();
  } catch (error) {
    if (error instanceof TypeError || controller.signal.aborted) throw new Error("服务器不可用");
    throw error;
  } finally { clearTimeout(timer); }
}

function formatApiError(status: number, body: string, statusText: string): string {
  if (status === 413) {
    return "文件过大，请上传 50MB 以内的音频。";
  }
  if (status === 429) {
    return "请求太频繁，请稍后再试。";
  }
  if (status === 400) {
    return "文件类型或内容不支持，请换一个音频文件重试。";
  }
  if (status >= 500) {
    return "服务器不可用";
  }

  try {
    const parsed = JSON.parse(body);
    if (typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // Fall through to the generic message.
  }

  return statusText || "请求失败，请稍后再试。";
}

export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(`${APP_BASE}/health`);
}

export async function listProducers(): Promise<ProducerListResponse> {
  return request<ProducerListResponse>(`${APP_BASE}/catalog/producers.json`);
}

export async function getProducer(slug: string): Promise<ProducerInfo> {
  return request<ProducerInfo>(`${APP_BASE}/catalog/producers/${encodeURIComponent(slug)}.json`);
}

export async function analyzeAudio(
  file: File,
  onProgress?: (pct: number) => void
): Promise<AnalyzeResponse> {
  return uploadAudio<AnalyzeResponse>(`${APP_BASE}/api/analyze`, file, onProgress);
}

export async function createAnalyzeJob(
  file: File,
  onProgress?: (pct: number) => void
): Promise<JobStatusResponse> {
  const health = await checkHealth();
  if (health.status !== "ok") throw new Error("服务器不可用");
  return uploadAudio<JobStatusResponse>(`${APP_BASE}/api/analyze/jobs`, file, onProgress);
}

export async function getAnalyzeJob(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`${APP_BASE}/api/jobs/${jobId}`);
}

function uploadAudio<T>(
  url: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.upload.addEventListener("load", () => {
      onProgress?.(100);
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("Invalid response"));
        }
      } else {
        reject(new Error(formatApiError(xhr.status, xhr.responseText, xhr.statusText)));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("服务器不可用")));
    xhr.addEventListener("timeout", () => reject(new Error("服务器不可用")));

    xhr.open("POST", url);
    xhr.timeout = 120000;
    xhr.send(formData);
  });
}
