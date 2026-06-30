import { useState } from 'react'
import SectionHeader from './SectionHeader'
import ModulePill from './ModulePill'
import Badge from './Badge'
import { C } from './constants'

function TestRow({ test, affectedModules }) {
  const [open, setOpen] = useState(false)
  const score = test.vrtq_composite
  const barColor = score > 0.8 ? C.red : score > 0.6 ? C.amber : C.teal

  return (
    <div style={{ borderBottom: `1px solid ${C.border}`,
      background: open ? C.surface : 'transparent' }}>
      <div onClick={() => setOpen(!open)} style={{
        display: 'grid', gridTemplateColumns: '28px 80px 1fr 100px 160px 24px',
        gap: 10, padding: '10px 14px', cursor: 'pointer', alignItems: 'center', fontSize: 12,
      }}>
        <span style={{ color: C.textMut, fontWeight: 600 }}>#{test.rank}</span>
        <span style={{ color: C.accent, fontFamily: 'monospace', fontSize: 11 }}>{test.test_id}</span>
        <ModulePill name={test.module} highlighted={affectedModules.includes(test.module)} />
        <Badge type={test.test_type} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ flex: 1, height: 3, background: C.border, borderRadius: 2 }}>
            <div style={{ width: `${(score * 100).toFixed(0)}%`, height: '100%',
              background: barColor, borderRadius: 2, transition: 'width 0.5s' }} />
          </div>
          <span style={{ color: C.textSec, fontSize: 11, minWidth: 36 }}>{score.toFixed(3)}</span>
        </div>
        <span style={{ color: open ? C.accent : C.textMut, fontSize: 10 }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ padding: '0 14px 12px 42px', fontSize: 11, color: C.textSec, lineHeight: 1.8 }}>
          <div><span style={{ color: C.textMut }}>Rationale: </span>{test.rationale}</div>
          <div><span style={{ color: C.textMut }}>Exec time: </span>{test.execution_time_seconds}s</div>
          <div style={{ marginTop: 4, display: 'flex', gap: 16, fontFamily: 'monospace', fontSize: 10 }}>
            <span style={{ color: C.accent }}>V</span>
            <span style={{ color: C.amber }}>R</span>
            <span style={{ color: C.teal }}>T</span>
            <span style={{ color: C.purple }}>Q</span>
            <span style={{ color: C.textMut }}>composite={score.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default function TestQueueTab({ report }) {
  const affected = report.git_context?.modules_affected || []
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 14 }}>
        <SectionHeader>
          Prioritized queue — top {report.top_tests.length} of {report.metrics.tests_run} selected
        </SectionHeader>
        <span style={{ fontSize: 11, color: C.textMut }}>click row to expand</span>
      </div>

      <div style={{ background: C.card, border: `1px solid ${C.border}`,
        borderRadius: 12, overflow: 'hidden' }}>
        <div style={{ display: 'grid',
          gridTemplateColumns: '28px 80px 1fr 100px 160px 24px',
          gap: 10, padding: '8px 14px', borderBottom: `1px solid ${C.border}`,
          fontSize: 10, color: C.textMut, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          <span>#</span><span>Test ID</span><span>Module</span>
          <span>Type</span><span>VRTQ Score</span><span />
        </div>
        {report.top_tests.map(t => (
          <TestRow key={t.test_id} test={t} affectedModules={affected} />
        ))}
      </div>

      <div style={{ marginTop: 10, fontSize: 11, color: C.textMut }}>
        Showing top {report.top_tests.length} ·{' '}
        {report.metrics.tests_run - report.top_tests.length} more selected ·{' '}
        full list via{' '}
        <code style={{ color: C.accent }}>python -m agents.orchestrator --save</code>
      </div>
    </div>
  )
}
