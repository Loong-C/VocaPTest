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
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(formatApiError(res.status, body, res.statusText));
  }
  return res.json();
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
    return "服务暂时无法完成分析，请稍后再试。";
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
  return request<HealthResponse>("/health");
}

export async function listProducers(): Promise<ProducerListResponse> {
  return request<ProducerListResponse>(`${APP_BASE}/api/producers`);
}

export async function getProducer(slug: string): Promise<ProducerInfo> {
  return request<ProducerInfo>(`${APP_BASE}/api/producers/${slug}`);
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

    xhr.addEventListener("error", () => reject(new Error("Network error")));

    xhr.open("POST", url);
    xhr.send(formData);
  });
}
