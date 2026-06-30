import { C } from './constants'

export default function SectionHeader({ children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
      <div style={{ width: 3, height: 16, background: C.accent, borderRadius: 2 }} />
      <span style={{ fontSize: 12, fontWeight: 600, color: C.textSec,
        textTransform: 'uppercase', letterSpacing: '0.08em' }}>{children}</span>
    </div>
  )
}
