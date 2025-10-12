import React from "react";

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { err: null };
    }
    static getDerivedStateFromError(error) {
        return { err: error };
    }
    componentDidCatch(error, info) {
        console.error("ErrorBoundary caught:", error, info);
    }
    render() {
        const { err } = this.state;
        if (err) {
            return (
                <div style={{ padding: 16, color: "#f44336" }}>
                    Something went wrong. Please reload.
                </div>
            );
        }
        return this.props.children;
    }
}
