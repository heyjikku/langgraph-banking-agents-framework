import React from 'react';
import Bar from './Bar';
import { money, pct } from '../lib/format';

export default function Overview({ customer: c, run: r }) {
  const ch = r.results.churn;
  const nba = r.results.nba;
  const dq = r.data_quality;
  const maxAbs = ch ? Math.max(...ch.attributions.map((a) => Math.abs(a.contribution)), 0.05) : 1;
  const spend = Object.entries(c.spend_by_category || {}).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const spendMax = Math.max(...spend.map((x) => x[1]), 1);

  return (
    <div className="grid">
      {ch && <div className="sec">
        <h3>Why churn is {ch.risk_level.toLowerCase()} <small>{pct(ch.churn_probability)} in 90 days · {pct(ch.churn_probability_30d)} in 30</small></h3>
        {ch.attributions.map((a) => <Bar key={a.feature} label={a.feature} value={a.contribution} max={maxAbs} />)}
        <p className="note">Points added to or removed from a 12% base rate. Sentiment sources agree {ch.sentiment_reconciliation.agreement == null ? 'n/a' : pct(ch.sentiment_reconciliation.agreement)}.</p>
        <div className="kv">
          <span>Play</span><b>{ch.recommended_action.action} · within {ch.recommended_action.urgency.toLowerCase()}</b>
          <span>Owner</span><b>{ch.recommended_action.owner} via {ch.recommended_action.channel}</b>
          <span>Offer</span><b style={{ fontWeight: 400 }}>{ch.recommended_action.offer}</b>
        </div>
      </div>}

      {nba && <div className="sec">
        <h3>
          Next best actions
          <small>{nba.suppressed ? 'suppressed' : `${nba.eligible_count} eligible of ${nba.candidates_considered}`} · fraud {nba.cross_agent_inputs.fraud} · credit {nba.cross_agent_inputs.credit}</small>
        </h3>
        {nba.suppressed ? (
          <div className="empty t-red">{nba.top_action.rationale}. Clear the hold, then re-run.</div>
        ) : nba.all_actions.length ? nba.all_actions.map((a, i) => (
          <div className="action" key={a.product_id}>
            <div>
              <div className="t">{a.product_name} <span className="badge slate">{a.category}</span> <span className={`badge ${a.urgency === 'High' ? 'red' : a.urgency === 'Medium' ? 'amber' : 'slate'}`}>{a.urgency}</span></div>
              <div className="m">{a.rationale} · via {a.delivery_channel}</div>
            </div>
            <div className="s">{pct(a.propensity_score)}<div className="m">{money(a.expected_revenue)}/yr</div></div>
            {i === 0 && a.llm_pitch && <div className="pitch">{a.llm_pitch}</div>}
          </div>
        )) : (
          <div className="empty">Nothing eligible after holding, score, income and asset filters.</div>
        )}
        <p className="note">Holds {nba.held_products.length ? nba.held_products.join(', ') : 'no mapped products'}.</p>
      </div>}

      <div className="sec">
        <h3>Balances <small>after quality gate</small></h3>
        <table><tbody>
          {c.accounts.map((a) => (
            <tr key={a.id}><td>{a.type}</td><td className="m">{a.status}</td><td className="num" style={{ color: a.balance < 0 ? 'var(--red)' : 'inherit' }}>{money(a.balance)}</td></tr>
          ))}
        </tbody></table>
        <div className="kv" style={{ marginTop: 10 }}>
          <span>Assets · liabilities</span><b>{money(c.assets)} · {money(c.liabilities)}</b>
          <span>Net position</span><b style={{ color: c.net < 0 ? 'var(--red)' : 'inherit' }}>{money(c.net)}</b>
          {c.salary_credits > 0 && <><span>Salary credits observed</span><b>{money(c.salary_credits)}</b></>}
        </div>
      </div>

      <div className="sec">
        <h3>Where the money goes <small>debits by category</small></h3>
        {spend.length ? spend.map(([k, v]) => <Bar key={k} label={k} value={v} max={spendMax} plain format={money} />) : <div className="empty">No debit history.</div>}
        <h3 style={{ marginTop: 18 }}>Data the agents trusted <small>trust {dq.trust_score.toFixed(2)}</small></h3>
        <div className="kv">
          <span>Repaired · excluded · high severity</span><b>{dq.fields_repaired} · {dq.records_excluded} · {dq.high_severity}</b>
          {dq.issues.filter((i) => i.severity === 'high').slice(0, 2).map((i, k) => <span key={k} style={{ gridColumn: '1 / -1', fontSize: 12 }}>{i.issue}</span>)}
        </div>
      </div>
    </div>
  );
}
