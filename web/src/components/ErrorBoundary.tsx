import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Unhandled UI error", error, errorInfo);
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="flex min-h-screen items-center justify-center bg-cream px-4 text-center text-text">
        <div className="card max-w-md p-8">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-red-50 text-red-400">
            <AlertTriangle size={28} />
          </div>
          <h1 className="mb-2 font-display text-2xl">页面暂时出错</h1>
          <p className="mb-5 text-sm leading-relaxed text-text-light">
            可能是页面组件加载失败，刷新后通常就能恢复。
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="btn-secondary"
          >
            <RefreshCw size={16} />
            刷新页面
          </button>
        </div>
      </div>
    );
  }
}
