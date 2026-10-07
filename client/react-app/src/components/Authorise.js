import React, { useState } from 'react';
import { api } from '../services/api';
import { TONE } from '../lib/format';

const EMPTY = { amount: '', merchant: '', merchant_category: '', channel: 'Online', hour_of_day: new Date().getHours(), is_international: false, transaction_count_1h: 1 };

/** Scores an in-flight transaction against this customer's baseline via the Fraud agent. */
export default function Authorise({ customerId }) {
  const [form, setForm] = useState(EMPTY);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const set = (key) => (e) => {
    setForm({ ...form, [key]: e.target.type === 'checkbox' ? e.target.checked : e.target.value });
    setError('');
  };

  const run = async () => {
    if (!form.amount || Number.isNaN(+form.amount) || +form.amount <= 0) { setError('Enter an amount greater than 0'); return; }
    setBusy(true); setError('');
    try {
      setResult(await api.authorise(customerId, {
        ...form, amount: +form.amount, hour_of_day: +form.hour_of_day, transaction_count_1h: +form.transaction_count_1h, transaction_id: `T-${Date.now()}`,
      }));
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  return (
    <div className="auth">
      <div className="row">
        <label>Amount<input type="number" value={form.amount} onChange={set('amount')} placeholder="0.00" /></label>
        <label>Merchant<input value={form.merchant} onChange={set('merchant')} placeholder="e.g. Costco" /></label>
        <label>Category<input value={form.merchant_category} onChange={set('merchant_category')} placeholder="e.g. Gift Cards" /></label>
      </div>
      <div className="row">
        <label>Channel
          <select value={form.channel} onChange={set('channel')}>
            {['Online', 'POS Terminal', 'ATM', 'Mobile App', 'Wire'].map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>
        <label>Hour<input type="number" min="0" max="23" value={form.hour_of_day} onChange={set('hour_of_day')} /></label>
        <label>Txns in last hour<input type="number" min="0" value={form.transaction_count_1h} onChange={set('transaction_count_1h')} /></label>
        <label className="chk"><input type="checkbox" checked={form.is_international} onChange={set('is_international')} /> International</label>
      </div>
      <div className="row">
        <button className="btn" onClick={run} disabled={busy}>{busy ? 'Scoring…' : 'Score this transaction'}</button>
        {error && <span className="err">{error}</span>}
      </div>
      {result && (
        <div className={`verdict ${TONE[result.decision]}`}>
          <div className={`d t-${TONE[result.decision]}`}>{result.decision}</div>
          <div>score {result.risk_score} · {result.latency_ms} ms</div>
          <div className="m">
            {result.inflight_signals.length ? result.inflight_signals.join(' · ') : 'no in-flight anomalies'}
            {result.rules_fired.length ? ` · ${result.rules_fired.join('; ')}` : ''}
          </div>
        </div>
      )}
    </div>
  );
}
