import React, { memo, useMemo, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Avatar from "@mui/material/Avatar";
import Tooltip from "@mui/material/Tooltip";
import IconButton from "@mui/material/IconButton";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import ReplayIcon from "@mui/icons-material/Replay";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import CircularProgress from "@mui/material/CircularProgress";
import ProgressiveText from "./ProgressiveText";
import Markdown from "../utils/markdown";

function getAvatarSrc(msg, isUser) {
  if (isUser) return msg.avatar || undefined;
  return msg.avatar || "/annie.jpg"; // Or wherever your bot image is in public/
}

function MessageBubbleBase({
  msg,
  onRetry,
  mode = "dark",
  showAvatar = true,
  typingEffect = true,
  streamedText,
  scrollToBottom,
  isTyping = false,
}) {
  const [copied, setCopied] = useState(false);
  const isUser = msg.role === "user";
  const bubbleBg = isUser ? (mode === "dark" ? "#1f1f1f" : "#f2f4f8") : (mode === "dark" ? "#141414" : "#fff");
  const textColor = mode === "dark" ? "#eaeaea" : "#111";
  const meta = useMemo(() => {
    try {
      return new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return "";
    }
  }, [msg.timestamp]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(msg.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {}
  };
  const avatarSrc = getAvatarSrc(msg, isUser);

  // Only bot and currently typing message gets glow
  const isAssistantTyping = !isUser && typingEffect && isTyping;

  return (
    <Box sx={{
      display: "flex",
      gap: 1.2,
      my: 1,
      alignItems: "flex-start",
      flexDirection: isUser ? "row-reverse" : "row"
    }}>
      {showAvatar && (
        <Avatar
          alt={isUser ? "You" : "Annie"}
          src={avatarSrc}
          sx={{
            width: 30,
            height: 30,
            bgcolor: isUser ? "#1976d2" : (mode === "dark" ? "#232323" : "#eef2ff"),
            color: isUser ? "#fff" : "#3b82f6",
            fontWeight: 700,
            mt: 0.2,
            userSelect: "none",
            boxShadow: isAssistantTyping
              ? "0 0 2px 2px #aeefff77, 0 0 8px 2px #d0eaff55"
              : "none",
            animation: isAssistantTyping
              ? "avatar-glow-rotate 2.2s linear infinite"
              : "none"
          }}
        >
          {(!avatarSrc) && (isUser ? "U" : "A")}
        </Avatar>
      )}
      <Box sx={{ maxWidth: { min: "320px", xs: "90vw" } }}>
        <Box sx={{
          background: bubbleBg,
          border: "1px solid",
          borderColor: "divider",
          color: textColor,
          borderRadius: 1.5,
          px: 1.5,
          py: 1.25,
          minHeight: 32,
          display: "flex",
          alignItems: "center"
        }}>
          <Typography component="div" sx={{ color: textColor, fontSize: "0.95rem", lineHeight: 1.65 }}>
            {msg.format === "markdown" ? (
              <Markdown>{msg.text}</Markdown>
            ) : isAssistantTyping ? (
              // Avatar + loading animation
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <CircularProgress size={18} sx={{ ml: 1, mr: 1 }} />
                <span style={{ opacity: 0.7, fontStyle: "italic" }}>Typing…</span>
              </span>
            ) : typingEffect ? (
              <ProgressiveText
                text={msg.text}
                active={typingEffect}
                speed={14}
                streamedValue={streamedText}
                onUpdate={scrollToBottom}
              />
            ) : (
              <ProgressiveText text={msg.text} active={false} />
            )}
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.75, opacity: 0.7 }}>
          {msg.status === "error" && <ErrorOutlineIcon sx={{ fontSize: 16, color: "#f44336" }} />}
          <Tooltip title={new Date(msg.timestamp).toLocaleString()}>
            <Typography variant="caption" sx={{ color: textColor, cursor: "pointer" }}>
              {meta}
            </Typography>
          </Tooltip>
          <Tooltip title={copied ? "Copied!" : "Copy"}>
            <IconButton size="small" onClick={copy} color="primary">
              <ContentCopyIcon fontSize="inherit" />
            </IconButton>
          </Tooltip>
          {msg.status === "error" && (
            <IconButton size="small" onClick={() => onRetry?.(msg)} color="primary">
              <ReplayIcon fontSize="inherit" />
            </IconButton>
          )}
        </Box>
      </Box>
    </Box>
  );
}

export default memo(MessageBubbleBase);
