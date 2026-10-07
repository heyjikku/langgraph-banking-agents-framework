import React, { useCallback, useEffect, useState } from 'react';
import CustomerRail from './components/CustomerRail';
import Portfolio from './components/Portfolio';
import Customer from './components/Customer';
import ErrorBoundary from './components/ErrorBoundary';
import { api, setApiKey } from './services/api';

export default function App() {
  const [health, setHealth] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [status, setStatus] = useState('Connecting…');
  const [needKey, setNeedKey] = useState(false);
  const [keyInput, setKeyInput] = useState('');

  const boot = useCallback(async () => {
    setNeedKey(false);
    try {
      setHealth(await api.health());
      const cs = await api.customers(40);
      setCustomers(cs.customers);
      setStatus('Executing portfolio…');
      const p = await api.portfolio(40);
      setPortfolio(p);
      setStatus(`${p.summary.customers} pipelines · ${p.summary.executed_ms.toLocaleString()} ms`);
    } catch (e) {
      if (e.auth) { setNeedKey(true); setStatus('API key required'); }
      else setStatus(`Cannot reach agent service — ${e.message}`);
    }
  }, []);

  useEffect(() => { boot(); }, [boot]);

  const onAuthError = useCallback(() => { setNeedKey(true); setStatus('API key required'); }, []);

  if (needKey) {
    return (
      <div className="gatekey">
        <h1>Customer 360</h1>
        <p>This service requires an API key.</p>
        <input type="password" value={keyInput} onChange={(e) => setKeyInput(e.target.value)} placeholder="X-API-Key" />
        <button className="btn" onClick={() => { setApiKey(keyInput); boot(); }}>Connect</button>
      </div>
    );
  }

  const railRows = portfolio ? portfolio.rows : customers.map((c) => ({ ...c, health: null }));

  return (
    <div className="app">
      <header className="top">
        <div className="brand"><b>{health ? health.bank_name : 'Customer 360'}</b><span>Customer 360</span></div>
        <div className="meta">
          <span>{health && <><em>{health.agents.length} agents</em> · {health.bank_profile} · {health.llm === 'claude' ? 'Claude narratives' : 'template narratives'}</>}</span>
          <span className="status">{status}</span>
        </div>
      </header>
      <div className="body">
        <CustomerRail rows={railRows} portfolioCount={portfolio && portfolio.summary.customers} selectedId={selectedId} onSelect={setSelectedId} />
        <main className="main">
          <ErrorBoundary resetKey={selectedId}>
            {selectedId === null
              ? <Portfolio data={portfolio} status={status} onPick={setSelectedId} />
              : <Customer id={selectedId} onStatus={setStatus} onAuthError={onAuthError} />}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
