import React from 'react';
import { money, healthColor } from '../lib/format';

export default function CustomerRail({ rows, portfolioCount, selectedId, onSelect }) {
  return (
    <nav className="rail" aria-label="Customers">
      <button className={`cust pf${selectedId === null ? ' on' : ''}`} onClick={() => onSelect(null)}>
        <span className="dot" style={{ background: 'var(--bank)' }} />
        <span><div className="nm">Portfolio</div><div className="sg">{portfolioCount != null ? `${portfolioCount} customers` : '…'}</div></span>
      </button>
      <div className="list">
        {rows.map((x) => (
          <button key={x.customer_id} className={`cust${x.customer_id === selectedId ? ' on' : ''}`} onClick={() => onSelect(x.customer_id)} aria-pressed={x.customer_id === selectedId}>
            <span className="dot" style={{ background: healthColor(x.health) }} />
            <span><div className="nm">{x.name}</div><div className="sg">{x.segment}{x.value_at_risk ? ` · ${money(x.value_at_risk)} at risk` : ''}</div></span>
            <span className="hs">{x.health == null ? '' : x.health.toFixed(0)}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
