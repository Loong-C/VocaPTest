import { Loader2 } from "lucide-react";

interface Props {
  size?: "sm" | "md" | "lg";
  text?: string;
  className?: string;
}

const sizeMap = {
  sm: "w-5 h-5",
  md: "w-8 h-8",
  lg: "w-12 h-12",
};

export default function LoadingSpinner({ size = "md", text, className = "" }: Props) {
  return (
    <div className={`flex flex-col items-center gap-3 ${className}`}>
      <div className="relative">
        <Loader2
          className={`${sizeMap[size]} text-pink animate-spin`}
        />
        <div
          className={`absolute inset-0 ${sizeMap[size]} rounded-full
                      bg-gradient-to-r from-pink to-purple opacity-20 blur-md animate-pulse`}
        />
      </div>
      {text && (
        <p className="text-text-light text-sm font-medium animate-pulse">{text}</p>
      )}
    </div>
  );
}
