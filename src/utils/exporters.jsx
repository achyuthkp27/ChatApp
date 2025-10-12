export function exportJSON(messages, filename = "chat.json") {
  const data = JSON.stringify(messages, null, 2);
  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  trigger(url, filename);
}

export function exportMarkdown(messages, filename = "chat.md") {
  const lines = messages.map((m) => {
    const who = m.role === "user" ? "User" : "Annie";
    const time = new Date(m.timestamp).toLocaleString();
    return `### ${who} -  ${time}\n\n${m.text}\n`;
  });
  const blob = new Blob([lines.join("\n---\n\n")], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  trigger(url, filename);
}

function trigger(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}