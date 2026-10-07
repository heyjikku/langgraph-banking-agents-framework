import React from 'react';

/** Signed contribution bar (red right of centre, green left) or plain fill bar. */
export default function Bar({ label, value, max, plain = false, format }) {
  const width = `${Math.min(Math.abs(value) / max, 1) * (plain ? 100 : 50)}%`;
  const text = format ? format(value) : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}`;
  return (
    <div className="bar">
      <span>{label}</span>
      <div className="tr">
        <i className={plain ? 'plain' : value >= 0 ? 'pos' : 'neg'} style={{ width }} />
      </div>
      <span className="n">{text}</span>
    </div>
  );
}
