import React from 'react';

export default function Insight({ label, value, basis, tone }) {
  return (
    <div className="ins">
      <div className="l">{label}</div>
      <div className="v" style={{ color: tone || 'inherit' }}>{value}</div>
      <div className="b">{basis}</div>
    </div>
  );
}
