import React from 'react';
import { pct } from '../lib/format';

/** One agent's verdict: decision, the number behind it, one-line reason. */
export default function Tile({ agent, tone, decision, value, reason, confidence }) {
  return (
    <div className={`tile ${tone}`}>
      <div className="a">{agent}</div>
      <div className={`d t-${tone}`}>{decision}</div>
      <div className="v">{value}</div>
      <div className="r">{reason}</div>
      {confidence < 0.9 && <div className="c">confidence {pct(confidence)}</div>}
    </div>
  );
}
