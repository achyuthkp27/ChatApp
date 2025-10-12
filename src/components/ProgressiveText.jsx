import React, { useEffect, useRef, useState } from "react";

// Add any more punctuation you want to handle here
const TRAILING_PUNCT = ['.', ',', '!', '?', ';', ':'];

// Utility for splitting off trailing punctuation from links/emails
function splitTrailingPunct(str) {
  let cut = str.length;
  while (cut > 0 && TRAILING_PUNCT.includes(str[cut - 1])) cut--;
  if (cut === str.length) return [str, ""];
  return [str.slice(0, cut), str.slice(cut)];
}

// Complete progressive linkifier for URLs and emails, never includes trailing punctuation
function linkifyAll(text) {
  // Match all URLs and plain emails
  const urlOrEmailRe = /((https?:\/\/[^\s]+)|([\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}))/g;
  const result = [];
  let lastIdx = 0;
  let match;
  while ((match = urlOrEmailRe.exec(text))) {
    if (match.index > lastIdx) result.push(text.slice(lastIdx, match.index));
    // Which did we match?
    let matchedFull = match[0];
    let linkNode = null;
    const [clean, trailing] = splitTrailingPunct(matchedFull);
    if (clean.startsWith("http")) {
      linkNode = (
        <a
          key={match.index}
          href={clean}
          target="_blank"
          rel="noopener noreferrer nofollow"
          style={{
            color: "#8ab4f8",
            textDecoration: "underline",
            wordBreak: "break-all"
          }}
        >{clean}</a>
      );
    } else if (clean.includes("@")) {
      linkNode = (
        <a
          key={match.index}
          href={`mailto:${clean}`}
          style={{ color: "#8ab4f8", wordBreak: "break-word", textDecoration: "underline" }}
        >{clean}</a>
      );
    }
    result.push(
      <React.Fragment key={match.index}>
        {linkNode}
        {trailing}
      </React.Fragment>
    );
    lastIdx = match.index + matchedFull.length;
  }
  if (lastIdx < text.length) {
    result.push(text.slice(lastIdx));
  }
  return result;
}

export default function ProgressiveText({
  text,
  speed = 16,
  active = true,
  streamedValue,
  onDone,
  onUpdate
}) {
  const source = streamedValue ?? text ?? "";
  const [cursor, setCursor] = useState(active && !streamedValue ? 0 : source.length);
  const lastLenRef = useRef(source.length);

  useEffect(() => {
    if (!active) {
      setCursor(source.length);
      return;
    }
    if (cursor === source.length) {
      onDone?.();
      return;
    }
    const handle = setTimeout(() => setCursor(c => Math.min(c + 1, source.length)), speed);
    return () => clearTimeout(handle);
  }, [cursor, source, active, speed, streamedValue, onDone]);

  useEffect(() => {
    if (streamedValue != null && source.length !== lastLenRef.current) {
      lastLenRef.current = source.length;
      setCursor(source.length);
      onUpdate?.();
    }
  }, [streamedValue, source.length, onUpdate]);

  useEffect(() => {
    onUpdate?.();
  }, [cursor, onUpdate]);

  return (
    <span style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", overflowWrap: "anywhere" }}>
      {linkifyAll(source.slice(0, cursor))}
    </span>
  );
}
