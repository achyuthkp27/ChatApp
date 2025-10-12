import React from "react";
import { Box, TextField, IconButton, CircularProgress, Paper } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

export default function InputBar({ value, onChange, onSend, disabled, offline }) {
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !disabled) {
        onSend();
      }
    }
  };

  return (
    <Paper elevation={0} sx={{ width: "100%", maxWidth: "820px", borderRadius: 2, bgcolor: "background.paper", border: "1px solid", borderColor: "divider", overflow: "hidden" }}>
      <Box sx={{ display: "flex", alignItems: "center", p: 1 }}>
        <TextField
          fullWidth
          multiline
          minRows={1}
          maxRows={8}
          variant="outlined"
          size="small"
          placeholder={offline ? "Offline (will send when online)" : "Type a message..."}
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          inputProps={{ "aria-label": "Message input" }}
          InputProps={{ sx: { borderRadius: 2, "& fieldset": { borderColor: "divider !important" } } }}
        />
        <IconButton color="primary" onClick={onSend} aria-label="Send message" disabled={disabled || !value.trim()} sx={{ ml: 1 }}>
          {disabled ? <CircularProgress size={20} /> : <SendIcon />}
        </IconButton>
      </Box>
    </Paper>
  );
}
