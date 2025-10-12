import React, { useEffect, useState } from "react";
import { Dialog, DialogContent, TextField, List, ListItemButton, ListItemText, Divider } from "@mui/material";

const commands = [
  { key: "clear", label: "Clear conversation" },
  { key: "export-json", label: "Export as JSON" },
  { key: "export-md", label: "Export as Markdown" },
  { key: "help", label: "Show help" },
  { key: "theme-light", label: "Switch to Light Mode" },
  { key: "theme-dark", label: "Switch to Dark Mode" },
  { key: "font-larger", label: "Increase Font Size" },
  { key: "font-smaller", label: "Decrease Font Size" },
  { key: "focus-input", label: "Focus Message Input" },
  { key: "scroll-bottom", label: "Scroll to Latest Message" },
  { key: "scroll-top", label: "Scroll to First Message" },
  { key: "shortcuts", label: "Show Keyboard Shortcuts" },
  { key: "toggle-typing", label: "Toggle Typing Indicator" },
  { key: "mute-chat", label: "Mute Chat Notifications" },
  { key: "unmute-chat", label: "Unmute Chat Notifications" },
  { key: "accessibility", label: "Toggle High Contrast Mode" },
  { key: "search", label: "Search in Messages" },
  { key: "refresh-chat", label: "Refresh Messages" },
];

export default function CommandPalette({ open, onClose, onCommand }) {
  const [q, setQ] = useState("");

  useEffect(() => { if (!open) setQ(""); }, [open]);
  const filtered = commands.filter(cmd => cmd.label.toLowerCase().includes(q.toLowerCase()));

  const handleSelect = cmd => {
    onCommand?.(cmd.key);
    onClose?.();
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogContent>
        <TextField
          autoFocus fullWidth placeholder="Type a command"
          value={q} onChange={e => setQ(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter" && filtered.length > 0) {
              handleSelect(filtered[0]);
            }
          }}
        />
        <Divider sx={{ mt: 2, mb: 1 }} />
        <List>
          {filtered.map(cmd => (
            <ListItemButton key={cmd.key} onClick={() => handleSelect(cmd)}>
              <ListItemText primary={cmd.label} />
            </ListItemButton>
          ))}
        </List>
      </DialogContent>
    </Dialog>
  );
}
