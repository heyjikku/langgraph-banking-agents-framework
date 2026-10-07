import React from 'react';
import Bar from './Bar';
import Authorise from './Authorise';
import { money, pct, TONE } from '../lib/format';

export default function Risk({ customer: c, run: r }) {
  const f = r.results.fraud;
  const cr = r.results.credit_risk;
  const gates = !cr ? [] : [
    { ok: cr.probability_of_default <= 0.2, text: `PD ${pct(cr.probability_of_default, 1)} vs policy max 20%` },
    { ok: cr.delinquent_accounts < 3, text: `${cr.delinquent_accounts} delinquent accounts vs limit 3` },
    { ok: cr.debt_to_income <= 0.5, text: `DTI ${pct(cr.debt_to_income)} vs max 50%` },
    { ok: !f || f.recommendation !== 'Block', text: f ? `Fraud agent: ${f.recommendation}` : 'Fraud agent not enabled' },
  ];
  const pdComponents = cr ? Object.entries(cr.pd_components).filter(([, v]) => v !== 0) : [];
  const pdMax = Math.max(...pdComponents.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div className="grid">
      {f && <div className="sec">
        <h3>Fraud verdict <small>composite {f.composite_risk_score} · block ≥75 · review ≥45</small></h3>
        <div className="kv">
          <span>Alert pressure</span><b>{f.score_components.alert_pressure} <span className="m">× {f.score_components.weights.alert}</span></b>
          <span>Rules fired</span><b>{f.rules_fired.length ? f.rules_fired.join('; ') : 'none'}</b>
          <span>Open · resolved alerts</span><b>{f.alert_pressure.open_count} · {f.alert_pressure.resolved_count}</b>
        </div>
        {f.alert_details.length ? (
          <>
            <table style={{ marginTop: 12 }}>
              <thead><tr><th>Type</th><th>Severity</th><th>Status</th><th className="num">Weight</th></tr></thead>
              <tbody>
                {f.alert_details.map((a) => (
                  <tr key={a.id}>
                    <td>{a.type}</td>
                    <td><span className={`badge ${a.severity === 'Critical' ? 'red' : a.severity === 'High' ? 'amber' : 'slate'}`}>{a.severity}</span></td>
                    <td>{a.status}</td>
                    <td className="num">{a.weight_applied.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="note">Resolved and false-positive alerts keep 5% of their weight; open alerts keep 100%.</p>
          </>
        ) : <div className="empty">No fraud alerts on file.</div>}
        <h3 style={{ marginTop: 18 }}>Score a transaction now <small>live authorisation against this customer's baseline</small></h3>
        <Authorise customerId={c.customer_id} />
      </div>}

      {cr && <div className="sec">
        <h3>Credit decision <small>grade {cr.internal_grade} · {cr.bureau_band}</small></h3>
        <div className="gate">
          {gates.map((g, i) => (
            <React.Fragment key={i}><i className={g.ok ? 'p' : 'f'}>{g.ok ? '✓' : '✕'}</i><span>{g.text}</span></React.Fragment>
          ))}
        </div>
        <div className="kv" style={{ marginTop: 14 }}>
          <span>Appetite</span><b className={`t-${TONE[cr.risk_appetite]}`}>{cr.risk_appetite}</b>
          <span>Max eligible loan</span><b>{money(cr.max_eligible_loan)} <span className="m">{cr.income_multiple_cap}× income cap</span></b>
          <span>Indicative rate</span><b>{cr.recommended_rate}%</b>
          <span>Expected loss</span><b>{money(cr.expected_loss)} <span className="m">PD × LGD {cr.loss_given_default} × EAD {money(cr.exposure_at_default)}</span></b>
          {cr.score_reliability !== 'ok' && <><span>Bureau score</span><b className="t-amber">contradicted by delinquencies</b></>}
        </div>
        <h3 style={{ marginTop: 18 }}>What drives the PD <small>{pct(cr.probability_of_default, 1)}</small></h3>
        {pdComponents.map(([k, v]) => <Bar key={k} label={k.replace('_', ' ')} value={v} max={pdMax} />)}
        <h3 style={{ marginTop: 18 }}>Scorecard <small>of 100</small></h3>
        {Object.entries(cr.scorecard_breakdown).map(([k, v]) => <Bar key={k} label={k} value={v} max={35} plain format={(x) => x} />)}
      </div>}
    </div>
  );
}
