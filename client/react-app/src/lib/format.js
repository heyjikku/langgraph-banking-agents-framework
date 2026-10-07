export const money = (n) => (n == null ? '—' : `${n < 0 ? '−' : ''}$${Math.round(Math.abs(n)).toLocaleString()}`);
export const pct = (x, d = 0) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);

export const TONE = {
  Block: 'red', 'Flag for Review': 'amber', Allow: 'green',
  Decline: 'red', Monitor: 'amber', 'Within Limit': 'green',
  High: 'red', Medium: 'amber', Low: 'green', Critical: 'red',
};

export const healthColor = (h) => (h == null ? 'var(--mute)' : h > 68 ? 'var(--green)' : h > 52 ? 'var(--amber)' : 'var(--red)');
