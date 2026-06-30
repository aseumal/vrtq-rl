import { C } from './constants'

export default function ModulePill({ name, highlighted }) {
  return (
    <span style={{
      background: highlighted ? C.accentLo : C.card,
      color: highlighted ? C.accent : C.textSec,
      border: `1px solid ${highlighted ? C.accent : C.border}`,
      fontSize: 11, padding: '2px 8px', borderRadius: 99,
    }}>{name}</span>
  )
}
