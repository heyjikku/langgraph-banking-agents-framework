import React from 'react';

/** Keeps one broken panel from blanking the whole dashboard. */
export default class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidUpdate(prev) { if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null }); }
  render() {
    if (this.state.error) {
      return (
        <div className="loading t-red">
          This panel couldn't render: {String(this.state.error.message || this.state.error)}.
          <div className="m">Check the run's agents_failed list, or re-run the agents.</div>
        </div>
      );
    }
    return this.props.children;
  }
}
