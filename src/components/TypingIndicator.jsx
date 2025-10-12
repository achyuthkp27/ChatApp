import React from "react";
import { Box } from "@mui/material";
/**
* TypingIndicator
* - Accessible: announces politely to screen readers
* - Subtle: three bouncing dots
* - Theme-aware: uses MUI tokens for background/border
*
* Props:
* - name?: string — used for aria-label (default: "Annie")
*/
export default function TypingIndicator({ name = "Annie" }) {
    return (
        <Box
            role="status"
            aria-live="polite"
            aria-label={`${name} is typing`}
            sx={{
                display: "inline-flex",
                alignItems: "center",
                gap: 0.8,
                px: 1.3,
                py: 0.9,
                borderRadius: 2,
                bgcolor: "background.paper",
                border: "1px solid",
                borderColor: "divider",
                boxShadow: "none",
            }}
        >
            {[0, 1, 2].map((i) => (
                <Box
                    key={i}
                    sx={{
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        backgroundColor: "primary.main",
                        opacity: 0.75,
                        animation: "typingBounce 1.2s infinite",
                        animationDelay: `${i * 0.15}s`,
                        "@keyframes typingBounce": {
                            "0%, 80%, 100%": { transform: "translateY(0)", opacity: 0.45 },
                            "40%": { transform: "translateY(-4px)", opacity: 1 },
                        },
                    }}
                />
            ))}
        </Box>
    );
}