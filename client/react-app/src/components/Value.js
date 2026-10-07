import React from 'react';
import { money, pct } from '../lib/format';

export default function Value({ customer: c, run: r }) {
  const v = r.results.clv;
  const s = r.results.sentiment;
  const ms = v ? v.model_selection : null;
  const horizons = v ? [['1 year', v.clv_1_year], ['3 years', v.clv_3_year], ['5 years', v.clv_5_year]] : [];
  const max = Math.max(...horizons.map((h) => h[1]), 1);

  return (
    <div className="grid">
      {v && <div className="sec">
        <h3>Lifetime value <small>{v.tier} · {v.tier_strategy}</small></h3>
        <div className="horizon">
          {horizons.map(([label, n]) => (
            <div key={label}><b>{money(n)}</b><i style={{ height: `${(n / max) * 80}px` }} /><span>{label}</span></div>
          ))}
        </div>
        <div className="kv" style={{ marginTop: 14 }}>
          <span>Annual revenue</span><b>{money(v.annual_revenue_estimate)} <span className="m">{ms.primary_model.replace('_', ' ')}</span></b>
          <span>Deposit · lending · fee income</span><b>{money(ms.relationship.deposit_income)} · {money(ms.relationship.lending_income)} · {money(ms.relationship.fee_income)}</b>
          <span>Transaction model</span><b>{ms.bgnbd.used ? `${ms.bgnbd.frequency_per_year}/yr · p-alive ${ms.bgnbd.p_alive}` : `skipped — ${ms.bgnbd.reason}`}</b>
          <span>Wallet share</span><b>{v.wallet_share_pct}% <span className="m">{money(v.competitor_gap)} held elsewhere</span></b>
        </div>
        {c.life_events.length > 0 && (
          <>
            <h3 style={{ marginTop: 18 }}>Life events</h3>
            <table><tbody>
              {c.life_events.map((e, k) => <tr key={k}><td>{e.date}</td><td>{e.type}</td><td className="m">{e.source}</td><td className="num">{e.conf == null ? '' : pct(e.conf)}</td></tr>)}
            </tbody></table>
          </>
        )}
      </div>}

      {s && <div className="sec">
        <h3>Sentiment across sources <small>{s.overall_sentiment} · {s.sentiment_score.toFixed(2)}</small></h3>
        {Object.entries(s.source_scores).map(([k, val]) => (
          <div className="src" key={k}>
            <span>{k.replace('_', ' ')}</span>
            <div className="tr"><b style={{ width: `${val * 100}%` }} /><i style={{ left: `${s.sentiment_score * 100}%` }} /></div>
            <span className="n">{val.toFixed(2)}</span>
          </div>
        ))}
        <p className="note">
          {s.contradiction_note ? `${s.contradiction_note}.` : 'All sources agree.'}
          {s.label_text_agreement_pct != null && ` Dataset labels matched the feedback text ${s.label_text_agreement_pct}% of the time.`}
        </p>
        <div className="kv">
          <span>CSAT · resolution · open</span><b>{s.avg_csat == null ? '—' : s.avg_csat.toFixed(1)} / 5 · {s.resolution_rate == null ? '—' : pct(s.resolution_rate)} · {s.open_interactions}</b>
          <span>NPS</span><b>{s.nps_band} ({c.nps})</b>
          <span>Escalate</span><b className={s.requires_escalation ? 't-red' : 't-green'}>{s.requires_escalation ? 'yes' : 'no'}</b>
        </div>
        {c.interactions.length > 0 && (
          <>
            <h3 style={{ marginTop: 18 }}>Recent interactions</h3>
            <table><tbody>
              {c.interactions.slice(0, 6).map((i, k) => (
                <tr key={k}>
                  <td>{i.ts}</td><td>{i.intent}</td><td className="m">{i.channel}</td>
                  <td><span className={`badge ${i.res === 'Resolved' ? 'green' : i.res === 'Pending' || i.res === 'Escalated' ? 'amber' : 'slate'}`}>{i.res}</span></td>
                  <td className="num">{i.sat ?? '—'}</td>
                </tr>
              ))}
            </tbody></table>
          </>
        )}
      </div>}
    </div>
  );
}
