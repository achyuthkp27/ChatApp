import React from "react";
import { Box, Typography, Avatar, IconButton, Tooltip } from "@mui/material";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import DownloadIcon from "@mui/icons-material/Download";
import DataObjectIcon from "@mui/icons-material/DataObject";

export default function Header({ mode, onToggleMode, onOpenPalette, onExportJSON, onExportMD }) {
    return (
        <Box
            sx={{
                position: "sticky",
                top: 0,
                zIndex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 1,
                px: 2,
                py: 1.25,
                borderBottom: "1px solid #232323",
                background: "rgba(15,15,15,0.86)",
                backdropFilter: "saturate(180%) blur(10px)",
            }}
        >
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Avatar sx={{ width: 28, height: 28, bgcolor: "#232323", color: "#8ab4f8", fontSize: 14, fontWeight: 700 }}>
                    A
                </Avatar>
                <Typography variant="subtitle2" sx={{ color: "#eaeaea", fontWeight: 700 }}>
                    Annie
                </Typography>
                <Typography variant="caption" sx={{ color: "#aaa", ml: 1 }}>
                    Chat
                </Typography>
            </Box>

            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Tooltip title="Command palette (Ctrl/⌘+K)">
                    <IconButton onClick={onOpenPalette} size="small" color="primary">
                        <span style={{ fontWeight: 700 }}>⌘K</span>
                    </IconButton>
                </Tooltip>
                <Tooltip title="Export JSON">
                    <IconButton onClick={onExportJSON} size="small" color="primary">
                        <DataObjectIcon fontSize="small" />
                    </IconButton>
                </Tooltip>
                <Tooltip title="Export Markdown">
                    <IconButton onClick={onExportMD} size="small" color="primary">
                        <DownloadIcon fontSize="small" />
                    </IconButton>
                </Tooltip>
                <Tooltip title={mode === "dark" ? "Light mode" : "Dark mode"}>
                    <IconButton onClick={onToggleMode} size="small" color="primary">
                        {mode === "dark" ? <LightModeIcon /> : <DarkModeIcon />}
                    </IconButton>
                </Tooltip>
            </Box>
        </Box>
    );
}

