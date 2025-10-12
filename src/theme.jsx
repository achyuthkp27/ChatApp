import { createTheme } from "@mui/material/styles";

const darkBg = "#0f0f0f";
const lightBg = "#f6f7fb";

export const buildTheme = (mode = "dark") =>
    createTheme({
        palette: {
            mode,
            background: {
                default: mode === "dark" ? darkBg : lightBg,
                paper: mode === "dark" ? "#121212" : "#ffffff",
            },
            primary: { main: mode === "dark" ? "#8ab4f8" : "#3366ff" },
            text: {
                primary: mode === "dark" ? "#eaeaea" : "#111",
                secondary: mode === "dark" ? "#aaa" : "#555",
            },
            divider: mode === "dark" ? "#232323" : "#e5e7eb",
        },
        shape: { borderRadius: 16 },
        components: {
            MuiCssBaseline: {
                styleOverrides: {
                    "*:focus-visible": {
                        outline: "2px solid #8ab4f8",
                        outlineOffset: "2px",
                    },
                },
            },
            MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
        },
        typography: {
            fontFamily:
                '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"',
        },
    });

export const colorTokens = {
    dark: {
        bg: darkBg,
        bubbleUser: "#1f1f1f",
        bubbleAssistant: "#141414",
        divider: "#232323",
        link: "#8ab4f8",
    },
    light: {
        bg: lightBg,
        bubbleUser: "#f2f4f8",
        bubbleAssistant: "#ffffff",
        divider: "#e5e7eb",
        link: "#3366ff",
    },
};
