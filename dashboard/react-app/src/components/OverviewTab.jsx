import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from 'recharts'
import MetricCard from './MetricCard'
import SectionHeader from './SectionHeader'
import { C, pct } from './constants'

export default function OverviewTab({ report }) {
  const m = report.metrics
  const rs = report.risk_summary

  const radarData = [
    { axis: 'Value',   score: 0.72 },
    { axis: 'Risk',    score: rs.avg_risk_score },
    { axis: 'Time',    score: 0.65 },
    { axis: 'Quality', score: 0.60 },
  ]

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))',
        gap: 10, marginBottom: 24 }}>
        <MetricCard label="FDR @ 25%" value={pct(m.fdr_25)} accent={C.red}
          sub={`${Math.round(m.fdr_25 * m.total_faults)} faults early`} />
        <MetricCard label="FDR @ 50%" value={pct(m.fdr_50)} accent={C.amber}
          sub={`of ${m.total_faults} total`} />
        <MetricCard label="FDR @ 100%" value={pct(m.fdr_100)} accent={C.teal}
          sub={`${m.faults_found} found`} />
        <MetricCard label="First fault at" value={`#${m.ttff}`} accent={C.green}
          sub="test position" />
        <MetricCard label="Suite reduction" value={pct(m.tsr)} accent={C.purple}
          sub={`${m.total_tests - m.tests_run} tests skipped`} />
        <MetricCard label="Time saved"
          value={`${(m.estimated_time_saved_seconds / 60).toFixed(0)}m`}
          accent={C.accent} sub="vs full suite" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16 }}>
        <div style={{ background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 12, padding: 18 }}>
          <SectionHeader>AI Summary</SectionHeader>
          <p style={{ color: C.textSec, lineHeight: 1.8, margin: 0, fontSize: 13 }}>
            {report.summary}
          </p>
          <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {[
              ['Affected modules', rs.modules_affected.length],
              ['Affected tests', rs.n_affected_tests],
              ['High-risk tests', rs.high_risk_tests],
              ['Avg risk score', rs.avg_risk_score.toFixed(3)],
            ].map(([k, v]) => (
              <div key={k} style={{ background: C.surface, borderRadius: 8,
                padding: '9px 12px', border: `1px solid ${C.border}` }}>
                <div style={{ fontSize: 10, color: C.textMut, marginBottom: 2 }}>{k}</div>
                <div style={{ fontSize: 17, fontWeight: 600, color: C.textPri }}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 12, padding: 18 }}>
          <SectionHeader>VRTQ Score Profile</SectionHeader>
          <div style={{ fontSize: 12, color: C.textSec, marginBottom: 6 }}>
            Top test: <span style={{ color: C.accent }}>{report.top_tests[0]?.test_id}</span>
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={C.gridLine} />
              <PolarAngleAxis dataKey="axis" tick={{ fill: C.axisTick, fontSize: 10 }} />
              <Radar dataKey="score" stroke={C.accent} fill={C.accent}
                fillOpacity={0.15} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginTop: 6 }}>
            {[['Value','0.30'],['Risk','0.35'],['Time','0.20'],['Quality','0.15']].map(([k,w]) => (
              <div key={k} style={{ fontSize: 11, color: C.textSec,
                display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                <span>{k}</span><span style={{ color: C.textMut }}>w={w}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
