import React from 'react';
import Insight from './Insight';
import { money, pct, TONE, healthColor } from '../lib/format';

export default function Portfolio({ data, status, onPick }) {
  if (!data) return <div className="loading">{status}</div>;
  const s = data.summary;
  const segments = Object.entries(s.segments).sort((a, b) => b[1].value_at_risk - a[1].value_at_risk);

  return (
    <>
      <div className="hero">
        <div>
          <h1>{data.bank_name}</h1>
          <div className="sub">{s.customers} customers scored by all six agents · {s.executed_ms.toLocaleString()} ms</div>
        </div>
        <div className="health">
          <div className="n" style={{ color: healthColor(s.avg_health) }}>{s.avg_health}</div>
          <div className="g">average relationship health</div>
        </div>
      </div>

      <div className="insights">
        <Insight label="Lifetime value at risk" value={s.clv_enabled ? money(s.value_at_risk) : '—'} tone="var(--red)" basis={s.clv_enabled ? `3-year CLV × churn probability, across ${money(s.clv_3y_total)} total CLV` : 'CLV agent not enabled in this profile'} />
        <Insight label="Revenue opportunity" value={money(s.nba_revenue_opportunity)} tone="var(--green)" basis="annual revenue of eligible next-best actions" />
        <Insight label="High churn risk" value={s.high_churn} basis={`${pct(s.high_churn / s.customers)} of portfolio · above 60% in 90 days`} />
        <Insight label="Fraud holds" value={s.fraud_blocks + s.fraud_reviews} basis={`${s.fraud_blocks} blocked · ${s.fraud_reviews} flagged for review`} />
        <Insight label="Credit declines" value={s.credit_declines} basis="hard policy gates: PD, delinquencies, DTI, fraud" />
        <Insight label="Data trust" value={s.avg_trust.toFixed(2)} basis="average share of records that needed no repair" />
      </div>

      <div className="grid wide">
        <div className="sec">
          <h3>Act on these first <small>ranked by {s.clv_enabled ? 'value at risk' : 'churn probability'} · as of {data.as_of_date}</small></h3>
          <table className="hover">
            <thead>
              <tr><th>Customer</th><th>Segment</th><th className="num">Health</th><th className="num">Churn</th><th className="num">Value at risk</th><th>Holds</th><th>Next action</th></tr>
            </thead>
            <tbody>
              {data.rows.slice(0, 12).map((r) => (
                <tr key={r.customer_id} tabIndex={0} onClick={() => onPick(r.customer_id)} onKeyDown={(e) => e.key === 'Enter' && onPick(r.customer_id)}>
                  <td><b>{r.name}</b><div className="m">{r.customer_id}</div></td>
                  <td>{r.segment}</td>
                  <td className="num" style={{ color: healthColor(r.health), fontWeight: 600 }}>{r.health == null ? '—' : r.health.toFixed(0)}</td>
                  <td className="num"><span className={`badge ${TONE[r.churn_level] || 'slate'}`}>{pct(r.churn_probability)}</span></td>
                  <td className="num">{r.value_at_risk == null ? '—' : money(r.value_at_risk)}{r.clv_3y != null && <div className="m">of {money(r.clv_3y)}</div>}</td>
                  <td>
                    {r.fraud === 'Block' && <span className="badge red">Fraud block</span>}
                    {r.fraud === 'Flag for Review' && <span className="badge amber">Fraud review</span>}
                    {r.credit === 'Decline' && <span className="badge red">Credit decline</span>}
                    {r.escalate && <span className="badge amber">CX escalation</span>}
                  </td>
                  <td>
                    {r.nba_suppressed ? <span className="t-red">Clear fraud hold</span> : r.nba || '—'}
                    {!r.nba_suppressed && r.nba_revenue > 0 && <div className="m">{money(r.nba_revenue)}/yr eligible</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="sec">
          <h3>By segment</h3>
          <table>
            <thead><tr><th>Segment</th><th className="num">Customers</th><th className="num">High churn</th><th className="num">3-year CLV</th><th className="num">At risk</th></tr></thead>
            <tbody>
              {segments.map(([name, v]) => (
                <tr key={name}>
                  <td>{name}</td><td className="num">{v.n}</td><td className="num">{v.high_churn}</td>
                  <td className="num">{money(v.clv_3y)}</td><td className="num" style={{ color: 'var(--red)' }}>{money(v.value_at_risk)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">Value at risk concentrates where CLV is high and churn is above the bank's threshold — those relationships pay back an RM call first.</p>
        </div>
      </div>
    </>
  );
}
