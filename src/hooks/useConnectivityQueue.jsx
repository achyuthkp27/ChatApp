import { useEffect, useRef, useState } from "react";

export function useConnectivityQueue() {
  const [online, setOnline] = useState(
    typeof navigator === "undefined" ? true : navigator.onLine
  );
  const queueRef = useRef([]);

  useEffect(() => {
    const on = () => setOnline(true);
    const off = () => setOnline(false);
    window.addEventListener("online", on);
    window.addEventListener("offline", off);
    return () => {
      window.removeEventListener("online", on);
      window.removeEventListener("offline", off);
    };
  }, []);

  const enqueue = (fn) => {
    queueRef.current.push(fn);
  };

  const flush = async () => {
    while (queueRef.current.length) {
      const fn = queueRef.current.shift();
      try {
        // eslint-disable-next-line no-await-in-loop
        await fn();
      } catch {
        break;
      }
    }
  };

  useEffect(() => {
    if (online) flush();
  }, [online]);

  return { online, enqueue };
}
