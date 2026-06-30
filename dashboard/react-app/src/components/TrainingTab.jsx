import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer } from 'recharts'
import SectionHeader from './SectionHeader'
import { C, pct } from './constants'

const HYPERPARAMS = [
  ['Algorithm',      'PPO',  'Proximal Policy Optimization'],
  ['Total timesteps','100k', 'n_steps=2048, batch=64'],
  ['Discount γ',     '0.99', 'patient long-horizon agent'],
  ['Learning rate',  '3e-4', 'Adam optimizer'],
  ['Clip range ε',   '0.2',  'policy update bound'],
  ['Entropy coef',   '0.01', 'exploration bonus'],
]

export default function TrainingTab({ curveData, source, onRefresh, loading }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 14 }}>
        <SectionHeader>
          PPO Learning Curve
          {source === 'mlflow' && (
            <span style={{ marginLeft: 8, background: C.accentLo, color: C.accent,
              fontSize: 10, padding: '1px 6px', borderRadius: 99 }}>live</span>
          )}
          {source === 'simulated' && (
            <span style={{ marginLeft: 8, color: C.textMut, fontSize: 10 }}>simulated</span>
          )}
        </SectionHeader>
        <button onClick={onRefresh} disabled={loading} style={{
          padding: '5px 14px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: C.card, color: loading ? C.textMut : C.accent,
          fontSize: 12, cursor: loading ? 'default' : 'pointer',
        }}>
          {loading ? 'Loading…' : '↺ Refresh'}
        </button>
      </div>

      <div style={{ background: C.card, border: `1px solid ${C.border}`,
        borderRadius: 12, padding: 18, marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={curveData}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.gridLine} />
            <XAxis dataKey="step" tick={{ fill: C.axisTick, fontSize: 9 }}
              tickFormatter={v => `${v/1000}k`} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: C.axisTick, fontSize: 9 }} axisLine={false}
              tickLine={false} tickFormatter={pct} domain={[0, 0.9]} />
            <Tooltip contentStyle={{ background: C.tooltipBg, border: `1px solid ${C.tooltipBorder}`,
              borderRadius: 8, color: C.textPri, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
              formatter={v => [pct(v), 'FDR']} />
            <Line dataKey="ppo"    name="PPO (VRTQ-RL)"  stroke={C.teal}
              strokeWidth={2.5} dot={false} />
            <Line dataKey="vrtq"   name="VRTQ Heuristic" stroke={C.amber}
              strokeWidth={1.5} dot={false} strokeDasharray="5 3" />
            <Line dataKey="random" name="Random"         stroke={C.textMut}
              strokeWidth={1.5} dot={false} strokeDasharray="3 3" />
            <Legend wrapperStyle={{ fontSize: 11, color: C.textSec }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10 }}>
        {HYPERPARAMS.map(([label, value, sub]) => (
          <div key={label} style={{ background: C.card, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: '12px 14px' }}>
            <div style={{ fontSize: 10, color: C.textMut, marginBottom: 3 }}>{label}</div>
            <div style={{ fontSize: 18, fontWeight: 600, color: C.accent }}>{value}</div>
            <div style={{ fontSize: 10, color: C.textMut, marginTop: 2 }}>{sub}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
