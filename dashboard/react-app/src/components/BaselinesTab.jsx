import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer } from 'recharts'
import SectionHeader from './SectionHeader'
import { C, METHOD_COLOR, pct } from './constants'

export default function BaselinesTab({ comparison, onRefresh, loading }) {
  const chartData = comparison.map(d => ({
    ...d,
    'FDR@25%': +(d.fdr_25 * 100).toFixed(1),
    'FDR@50%': +(d.fdr_50 * 100).toFixed(1),
    'TSR%':    +(d.tsr * 100).toFixed(1),
  }))

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 14 }}>
        <SectionHeader>Method Comparison</SectionHeader>
        <button onClick={onRefresh} disabled={loading} style={{
          padding: '5px 14px', borderRadius: 6, border: `1px solid ${C.border}`,
          background: C.card, color: loading ? C.textMut : C.accent,
          fontSize: 12, cursor: loading ? 'default' : 'pointer',
        }}>
          {loading ? 'Running…' : '↺ Re-run'}
        </button>
      </div>

      <div style={{ background: C.card, border: `1px solid ${C.border}`,
        borderRadius: 12, padding: 18, marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={chartData} barCategoryGap="25%" barGap={3}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.gridLine} vertical={false} />
            <XAxis dataKey="method" tick={{ fill: C.axisTick, fontSize: 10 }}
              axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: C.axisTick, fontSize: 9 }} axisLine={false}
              tickLine={false} tickFormatter={v => `${v}%`} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: C.tooltipBg, border: `1px solid ${C.tooltipBorder}`,
              borderRadius: 8, color: C.textPri, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
              formatter={v => [`${v}%`]} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.textSec }} />
            <Bar dataKey="FDR@25%" fill={C.accent} radius={[3,3,0,0]} />
            <Bar dataKey="FDR@50%" fill={C.teal}   radius={[3,3,0,0]} />
            <Bar dataKey="TSR%"    fill={C.amber}   radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ background: C.card, border: `1px solid ${C.border}`,
        borderRadius: 12, overflow: 'hidden' }}>
        <div style={{ display: 'grid',
          gridTemplateColumns: '1fr 80px 80px 80px 60px 60px',
          gap: 10, padding: '7px 14px', borderBottom: `1px solid ${C.border}`,
          fontSize: 10, color: C.textMut, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          <span>Method</span><span>FDR@25%</span><span>FDR@50%</span>
          <span>FDR@100%</span><span>TTFF</span><span>TSR</span>
        </div>
        {comparison.map(row => (
          <div key={row.method} style={{ display: 'grid',
            gridTemplateColumns: '1fr 80px 80px 80px 60px 60px',
            gap: 10, padding: '11px 14px', borderBottom: `1px solid ${C.border}`,
            fontSize: 12 }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{ width: 9, height: 9, borderRadius: '50%', flexShrink: 0,
                background: METHOD_COLOR[row.method] || C.textMut }} />
              <span style={{ color: METHOD_COLOR[row.method] || C.textSec,
                fontWeight: row.method.includes('PPO') ? 600 : 400 }}>
                {row.method}
              </span>
            </span>
            {['fdr_25','fdr_50','fdr_100'].map(k => (
              <span key={k} style={{ color: row[k] === 0 ? C.textMut : C.textPri }}>
                {row[k] === 0 ? '—' : pct(row[k])}
              </span>
            ))}
            <span style={{ color: row.ttff === 0 ? C.textMut : C.textPri }}>
              {row.ttff === 0 ? '—' : `#${row.ttff}`}
            </span>
            <span style={{ color: row.tsr === 0 ? C.textMut : C.textPri }}>
              {row.tsr === 0 ? '—' : pct(row.tsr)}
            </span>
          </div>
        ))}
      </div>

      {comparison.some(row => row.note) && (
        <div style={{ marginTop: 10, fontSize: 11, color: C.textMut }}>
          {comparison.filter(row => row.note).map(row => (
            <div key={row.method}>{row.method}: {row.note}</div>
          ))}
        </div>
      )}
    </div>
  )
}
