import { C } from './constants'

const MAP = {
  unit: [C.accentLo, C.accent],
  integration: [C.tealBg, C.teal],
  e2e: [C.purpleBg, C.purple],
}

export default function Badge({ type }) {
  const [bg, color] = MAP[type] || [C.card, C.textSec]
  return (
    <span style={{ background: bg, color, fontSize: 10, padding: '2px 7px',
      borderRadius: 99, fontWeight: 500, fontFamily: 'monospace' }}>{type}</span>
  )
}
