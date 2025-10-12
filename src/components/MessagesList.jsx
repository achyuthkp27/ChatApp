import React from "react";
import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import TypingIndicator from "./TypingIndicator";

export default function MessagesList({ scrollerRef, children, typing, showJump, onJump }) {
  return (
    <Box
      ref={scrollerRef}
      role="log"
      aria-live="polite"
      sx={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        overflowY: "auto",
        px: { xs: 2, sm: 4 },
        py: 3,
        alignItems: "center"
      }}
    >
      <Box sx={{ width: "100%", maxWidth: "820px" }}>
        {children}
        {typing && (
          <Box sx={{ display: "flex", justifyContent: "flex-start", mt: 1 }}>
            <TypingIndicator />
          </Box>
        )}
        {showJump && (
          <Box sx={{ display: "flex", justifyContent: "center", mt: 1 }}>
            <Chip
              label="New messages"
              color="primary"
              size="small"
              onClick={onJump}
            />
          </Box>
        )}
      </Box>
    </Box>
  );
}
