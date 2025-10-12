import React from "react";
const urlRe = /(https?:\/\/[^\s]+)/;
const emailRe = /([\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,})/;
const phoneRe = /(\+?\d{1,4}[\s-]?\(?\d{3,5}\)?[\s-]?\d{3,4}[\s-]?\d{3,4})/;
const combined = new RegExp(`${urlRe.source}|${emailRe.source}|${phoneRe.source}`, "g");

export function linkify(text) {
  if (!text) return null;
  const parts = text.split(combined);
  return (
    <>
      {parts.map((part, idx) => {
        if (!part) return null;
        if (/^https?:\/\//.test(part)) {
          return (
            <a
              key={idx}
              href={part}
              target="_blank"
              rel="noopener noreferrer nofollow"
              style={{ color: "#8ab4f8", textDecoration: "underline", wordBreak: "break-all" }}
            >
              {part}
            </a>
          );
        }
        if (/^[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}$/.test(part)) {
          return (
            <a
              key={idx}
              href={`mailto:${part}`}
              style={{ color: "#8ab4f8", wordBreak: "break-word" }}
            >
              {part}
            </a>
          );
        }
        if (/^\+?\d/.test(part)) {
          return (
            <a
              key={idx}
              href={`tel:${part.replace(/\s+/g, "")}`}
              style={{ color: "#8ab4f8", wordBreak: "break-word" }}
            >
              {part}
            </a>
          );
        }
        return (
          <span key={idx} style={{ wordBreak: "break-word", overflowWrap: "anywhere" }}>
            {part}
          </span>
        );
      })}
    </>
  );
}
