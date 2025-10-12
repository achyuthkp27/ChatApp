import { useRef } from "react";

export function useTelemetry() {
  const tRef = useRef(null);
  const start = () => {
    tRef.current = performance.now();
  };
  const firstByte = () => {
    if (tRef.current != null) {
      const delta = performance.now() - tRef.current;
      console.log("[telemetry] TTFT(ms):", Math.round(delta));
    }
  };
  const end = (label = "total") => {
    if (tRef.current != null) {
      const delta = performance.now() - tRef.current;
      console.log(`[telemetry] ${label}(ms):`, Math.round(delta));
    }
    tRef.current = null;
  };
  return { start, firstByte, end };
}