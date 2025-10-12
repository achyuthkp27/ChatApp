import React, { useMemo } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { linkify } from "./linkify"; // <--- CORRECT way

marked.setOptions({ breaks: true, gfm: true });

const renderer = new marked.Renderer();
renderer.link = function(href, title, text) {
  return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
};

export function useSafeMarkdown(mdText) {
  return useMemo(() => {
    const inputWithLinks = linkify(mdText || "");
    const raw = marked.parse(inputWithLinks, { renderer });
    const clean = DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
    return clean;
  }, [mdText]);
}

const Markdown = ({ children }) => {
  const html = useSafeMarkdown(children || "");
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
};

export default Markdown;
