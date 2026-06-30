import { C } from './constants'

export default function MetricCard({ label, value, sub, accent }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
      padding: '14px 18px', position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', top: 0, left: 0, width: 3, height: '100%',
        background: accent || C.accent, borderRadius: '12px 0 0 12px' }} />
      <div style={{ fontSize: 10, color: C.textMut, textTransform: 'uppercase',
        letterSpacing: '0.08em', marginBottom: 5, paddingLeft: 4 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: accent || C.textPri,
        paddingLeft: 4, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.textSec, marginTop: 3, paddingLeft: 4 }}>{sub}</div>}
    </div>
  )
}
