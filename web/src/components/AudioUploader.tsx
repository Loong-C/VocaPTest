import { useDropzone } from "react-dropzone";
import { useCallback, useState } from "react";
import { Upload, Music, FileAudio, AlertCircle } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

const ALLOWED = {
  "audio/wav": [".wav"],
  "audio/mpeg": [".mp3"],
  "audio/flac": [".flac"],
  "audio/ogg": [".ogg"],
  "audio/mp4": [".m4a"],
  "audio/aac": [".aac"],
};

export default function AudioUploader({ onFile, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0 && accepted[0]) {
        onFile(accepted[0]);
      }
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: ALLOWED,
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
    disabled,
    onDragEnter: () => setDragOver(true),
    onDragLeave: () => setDragOver(false),
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-[var(--radius-card)]
          p-6 text-center cursor-pointer transition-all duration-300 sm:p-10
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
          ${
            isDragActive || dragOver
              ? "border-pink bg-pink/10 scale-[1.02]"
              : "border-pink-light/50 bg-white/40 hover:border-pink/60 hover:bg-pink/5"
          }
        `}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-3">
          <div
            className={`w-16 h-16 rounded-full flex items-center justify-center
                        transition-all duration-300
                        ${isDragActive ? "bg-pink/20 scale-110" : "bg-pink/10"}`}
          >
            {isDragActive ? (
              <Music className="w-8 h-8 text-pink animate-bounce" />
            ) : (
              <Upload className="w-8 h-8 text-pink-dark" />
            )}
          </div>

          <div>
            <p className="text-text font-medium">
              {isDragActive ? "✨ 松开以上传音频" : "拖拽音频文件到这里"}
            </p>
            <p className="mx-auto mt-1 max-w-xs text-sm leading-relaxed text-text-muted">
              或点击选择文件 · 支持 WAV、MP3、FLAC、OGG、M4A、AAC
            </p>
            <p className="text-text-muted text-xs mt-1">最大 50MB</p>
          </div>

          {/* Accepted formats hint */}
          <div className="flex flex-wrap justify-center gap-2 text-xs text-text-muted">
            <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-white/60 border border-pink-light/30">
              <FileAudio size={12} /> WAV
            </span>
            <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-white/60 border border-pink-light/30">
              <FileAudio size={12} /> MP3
            </span>
            <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-white/60 border border-pink-light/30">
              <FileAudio size={12} /> FLAC
            </span>
          </div>
        </div>
      </div>

      {/* Rejection feedback */}
      {fileRejections.length > 0 && (
        <div className="mt-3 flex items-start gap-2 p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
          <AlertCircle size={16} className="shrink-0 mt-0.5" />
          <div>
            {fileRejections[0]?.errors.map((e, i) => (
              <p key={i}>{e.message}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
