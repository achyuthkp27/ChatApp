import { useCallback, useEffect } from "react";

export function useAutoScroll(scrollerRef, deps = [], threshold = 80) {
  const isNearBottom = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  }, [scrollerRef, threshold]);

  const scrollToBottom = useCallback((smooth = true) => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({
      top: el.scrollHeight,
      behavior: smooth ? "smooth" : "auto"
    });
  }, [scrollerRef]);

  useEffect(() => {
    if (isNearBottom()) {
      requestAnimationFrame(() => scrollToBottom(true));
    }
    // eslint-disable-next-line
  }, [...deps, isNearBottom, scrollToBottom]);

  return { isNearBottom: isNearBottom(), scrollToBottom };
}
