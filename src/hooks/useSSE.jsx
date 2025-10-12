import { useEffect } from "react";

// Simple helper to start an EventSource and forward chunks.
// Returns a cleanup function to close the stream.
export function startSSE(url, params, { onChunk, onDone, onError }) {
  const fullUrl = new URL(url, window.location.origin);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v != null) fullUrl.searchParams.set(k, v);
  });
  const es = new EventSource(fullUrl.toString());
  es.onmessage = (e) => {
    if (e.data === "[DONE]") {
      onDone?.();
      es.close();
      return;
    }
    onChunk?.(e.data);
  };
  es.onerror = (err) => {
    onError?.(err);
    es.close();
  };
  return () => es.close();
}

// React-side convenience if needed (unused directly here)
export function useSSE(enabled, url, params, handlers) {
  useEffect(() => {
    if (!enabled) return undefined;
    const stop = startSSE(url, params, handlers || {});
    return () => stop?.();
  }, [enabled, url, JSON.stringify(params)]);
}
