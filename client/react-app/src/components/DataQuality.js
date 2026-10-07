import React from 'react';

export default function DataQuality({ run: r }) {
  const dq = r.data_quality;
  return (
    <div className="sec">
      <h3>Every repair, exclusion and flag the gate applied <small>trust {dq.trust_score.toFixed(3)}</small></h3>
      <div className="dq">
        <div><b>{dq.fields_repaired}</b>fields repaired</div>
        <div><b>{dq.records_excluded}</b>records excluded</div>
        <div><b>{dq.high_severity}</b>high severity</div>
      </div>
      {dq.issues.map((i, k) => (
        <div className="issue" key={k}>
          <span className={`badge ${i.severity === 'high' ? 'red' : i.severity === 'medium' ? 'amber' : 'slate'}`}>{i.severity}</span>
          <span className={`badge ${i.action === 'excluded' ? 'red' : i.action === 'repaired' ? 'bank' : 'slate'}`}>{i.action}</span>
          <span><span className="fld">{i.field}</span><br />{i.issue}</span>
        </div>
      ))}
      <p className="note">Trust = 1 − severity-weighted issues ÷ checks. Agents only ever see the cleaned payload.</p>
    </div>
  );
}
