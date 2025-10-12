import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, CssBaseline, ThemeProvider, Alert } from "@mui/material";
import axios from "axios";
import Header from "./components/Header";
import MessagesList from "./components/MessagesList";
import MessageBubble from "./components/MessageBubble";
import InputBar from "./components/InputBar";
import CommandPalette from "./components/CommandPalette";
import ErrorBoundary from "./components/ErrorBoundary";
import { buildTheme, colorTokens } from "./theme";
import { useAutoScroll } from "./hooks/useAutoScroll";
import { useConnectivityQueue } from "./hooks/useConnectivityQueue";
import { useTelemetry } from "./hooks/useTelemetry";
import { uid } from "./utils/id";
import { exportJSON, exportMarkdown } from "./utils/exporters";

const APIBASE = import.meta?.env?.VITE_API_BASE || "http://localhost:5000";
const STORAGEKEY = "annie-chat-messages-v3";
const DRAFTKEY = "annie-chat-draft-v1";
const STREAMENABLED = false;
const MAX_SIZE = 22;
const MIN_SIZE = 12;

export default function App() {
  const [mode, setMode] = useState(localStorage.getItem("theme-mode") || "dark");
  const theme = useMemo(() => buildTheme(mode), [mode]);
  const colors = mode === "dark" ? colorTokens.dark : colorTokens.light;

  const [messages, setMessages] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGEKEY)) || []; }
    catch { return []; }
  });

  const [input, setInput] = useState(localStorage.getItem(DRAFTKEY) || "");
  const [typing, setTyping] = useState(false);
  const [sending, setSending] = useState(false);
  const [showJump, setShowJump] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [streamBuffer, setStreamBuffer] = useState("");
  const [fontSize, setFontSize] = useState(16);
  const [muted, setMuted] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const scrollerRef = useRef(null);

  const { isNearBottom, scrollToBottom } = useAutoScroll(scrollerRef, [messages, typing]);
  const { online, enqueue } = useConnectivityQueue();
  const telemetry = useTelemetry();

  useEffect(() => { localStorage.setItem(STORAGEKEY, JSON.stringify(messages)); }, [messages]);
  useEffect(() => { localStorage.setItem(DRAFTKEY, input); }, [input]);
  useEffect(() => { localStorage.setItem("theme-mode", mode); }, [mode]);

  useEffect(() => {
    if (isNearBottom) setShowJump(false);
    else setShowJump(true);
  }, [messages, typing, isNearBottom]);

  const api = axios.create({
    baseURL: APIBASE,
    timeout: 20000,
  });

  const add = useCallback((m) => setMessages((prev) => [...prev, m]), []);
  const patch = useCallback((id, data) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...(typeof data === "function" ? data(m) : data) } : m))),
    []
  );

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    telemetry.start();
    const userMsg = { id: uid(), role: "user", text, timestamp: Date.now(), status: "sent", format: "text" };
    add(userMsg);
    setInput("");
    setTyping(true);

    const doSend = async () => {
      try {
        if (STREAMENABLED) {
          setStreamBuffer("");
          const botId = uid();
          add({ id: botId, role: "assistant", text: "", timestamp: Date.now(), status: "sending", format: "markdown" });
          await sendStream(text, botId);
        } else {
          const result = await sendPOST(text);
          const botMsg = { id: uid(), role: "assistant", text: result.text, timestamp: Date.now(), status: "sent", format: result.format };
          add(botMsg);
        }
        telemetry.end("total");
      } catch (e) {
        const errMsg = { id: uid(), role: "assistant", text: "Sorry, I couldn't reach the server. Tap retry.", timestamp: Date.now(), status: "error", format: "text" };
        add(errMsg);
      } finally {
        setTyping(false);
        setSending(false);
        requestAnimationFrame(() => scrollToBottom(true));
        setTimeout(() => {
          document.querySelector("textarea,input[aria-label='Message input']")?.focus();
        }, 0);
      }
    };
    if (online) await doSend();
    else enqueue(doSend);
  }, [input, sending, add, scrollToBottom, telemetry, online, enqueue]);

  const sendPOST = async (text) => {
    const res = await api.post("chat", { message: text });
    return {
      text: res.data?.answer,
      format: res.data?.format || "text",
    };
  };

  const sendStream = async (text, botId) => {
    patch(botId, { text: "Simulated streaming response", status: "sent", timestamp: Date.now() });
  };

  const handleRetry = useCallback(async (msg) => {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    try {
      patch(msg.id, { status: "sending" });
      const result = await sendPOST(lastUser.text);
      patch(msg.id, {
        text: result.text,
        status: "sent",
        format: result.format,
        timestamp: Date.now(),
      });
    } catch {
      patch(msg.id, { status: "error" });
    }
  }, [messages, patch]);

  const onCommand = useCallback((key) => {
    if (key === "clear") setMessages([]);
    else if (key === "export-json") exportJSON(messages);
    else if (key === "export-md") exportMarkdown(messages);
    else if (key === "help") add({
      id: uid(),
      role: "assistant",
      text: `
        Available commands:
        - clear
        - export-json
        - export-md
        - help
        - theme-light
        - theme-dark
        - font-larger
        - font-smaller
        - focus-input
        - scroll-bottom
        - scroll-top
        - shortcuts
        - toggle-typing
        - mute-chat
        - unmute-chat
        - accessibility
        - search
        - refresh-chat
      `,
      timestamp: Date.now(),
      status: "sent",
      format: "markdown"
    });
    else if (key === "theme-light") setMode("light");
    else if (key === "theme-dark") setMode("dark");
    else if (key === "font-larger") setFontSize((prev) => Math.min(prev + 1, MAX_SIZE));
    else if (key === "font-smaller") setFontSize((prev) => Math.max(prev - 1, MIN_SIZE));
    else if (key === "focus-input") document.querySelector("textarea,input[aria-label='Message input']")?.focus();
    else if (key === "scroll-bottom") scrollToBottom(true);
    else if (key === "scroll-top") scrollerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    else if (key === "shortcuts") setShowShortcuts(true);
    else if (key === "toggle-typing") setTyping((t) => !t);
    else if (key === "mute-chat") setMuted(true);
    else if (key === "unmute-chat") setMuted(false);
    else if (key === "accessibility") setHighContrast((h) => !h);
    else if (key === "search") setSearchOpen(true);
    else if (key === "refresh-chat") {
      setRefreshing(true);
      setTimeout(() => setRefreshing(false), 500);
    }
  }, [messages, add, setMode, setFontSize, scrollToBottom, scrollerRef, setTyping]);

  // Only animate the most recent assistant message
  const bubbles = useMemo(() =>
    messages.map((m, i) => (
      <MessageBubble
        key={m.id}
        msg={m}
        onRetry={handleRetry}
        mode={mode}
        typingEffect={i === messages.length - 1 && m.role === "assistant"}
        streamedText={undefined}
        scrollToBottom={() => scrollToBottom(true)}
      />
    )),
    [messages, handleRetry, mode, scrollToBottom, streamBuffer]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <Box sx={{ minHeight: "100vh", display: "flex", flexDirection: "column", bgcolor: colors.bg, fontSize, position: "relative" }}>
          {!online && <Alert severity="warning" sx={{ borderRadius: 0 }}>Offline: messages will send when reconnected.</Alert>}
          <Header mode={mode} onToggleMode={() => setMode((p) => (p === "dark" ? "light" : "dark"))} onOpenPalette={() => setPaletteOpen(true)} onExportJSON={() => exportJSON(messages)} onExportMD={() => exportMarkdown(messages)} />
          <Box sx={{ flex: 1, minHeight: 0, overflow: "hidden", pb: "80px", display: "flex", justifyContent: "center" }}>
            <MessagesList scrollerRef={scrollerRef} typing={typing} showJump={showJump} onJump={() => { scrollToBottom(true); setShowJump(false); }}>
              {bubbles}
            </MessagesList>
          </Box>
          <Box sx={{
            position: "fixed",
            left: 0,
            right: 0,
            bottom: 0,
            display: "flex",
            justifyContent: "center",
            zIndex: 20,
            bgcolor: "background.paper",
            pb: { xs: 1, sm: 2 }
          }}>
            <InputBar value={input} onChange={setInput} onSend={handleSend} disabled={sending} offline={!online} />
          </Box>
          <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} onCommand={onCommand} />
        </Box>
      </ErrorBoundary>
    </ThemeProvider>
  );
}
