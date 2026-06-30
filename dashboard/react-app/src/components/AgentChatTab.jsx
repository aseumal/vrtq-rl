import SectionHeader from './SectionHeader'
import { C } from './constants'

const ROLE_COLOR = {
  Supervisor: C.accent,
  Critic: C.purple,
  ToolExecutor: C.teal,
  Reviewer: C.textSec,
  RiskReviewer: C.amber,
  SLAReviewer: C.amber,
  ConfidenceReviewer: C.amber,
}

function MessageBubble({ msg }) {
  const color = ROLE_COLOR[msg.role] || C.textSec
  return (
    <div style={{ display: 'flex', gap: 10, padding: '10px 0', borderBottom: `1px solid ${C.border}` }}>
      <div style={{ width: 90, flexShrink: 0, fontSize: 11, fontWeight: 600, color }}>{msg.role}</div>
      <div style={{ flex: 1, fontSize: 12.5, color: C.textPri, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
        {msg.tool_call ? (
          <code style={{ background: C.surface, padding: '2px 6px', borderRadius: 4,
            color: C.teal, fontSize: 11.5, display: 'inline-block' }}>
            {msg.tool_call.name}({JSON.stringify(msg.tool_call.args)})
          </code>
        ) : msg.content}
      </div>
    </div>
  )
}

export default function AgentChatTab({ agenticStatus, onRun, result, loading }) {
  const available = !!agenticStatus?.openai_configured

  return (
    <div>
      <SectionHeader>Agentic Mode — Multi-Agent Conversation (AutoGen)</SectionHeader>

      <p style={{ color: C.textSec, fontSize: 12.5, lineHeight: 1.7, marginBottom: 14, maxWidth: 760 }}>
        A Supervisor agent drives the same pipeline stages as the default Overview tab via real tool/function
        calls — but now three reviewers can genuinely push back before the result is final: a RiskReviewer
        can challenge an ambiguous module classification before scoring even runs, an SLAReviewer can force a
        re-selection if execution time exceeds a CI budget, a ConfidenceReviewer can swap out a pick the RL
        policy was barely confident about, and a Critic reviews the final outcome quality — now constrained to
        justify any budget increase against the same time ceiling, not just ask for more.
      </p>

      {!available && (
        <div style={{ background: C.card, border: `1px solid ${C.amber}`, borderRadius: 10,
          padding: 14, marginBottom: 16, fontSize: 12.5, color: C.textSec }}>
          <strong style={{ color: C.amber }}>OPENAI_API_KEY</strong> is not configured (still the placeholder
          value). Set a real key in <code>.env</code> and restart the API to enable this mode.
        </div>
      )}

      <div style={{ display: 'flex', marginBottom: 16 }}>
        <button onClick={onRun} disabled={!available || loading} style={{
          padding: '7px 18px', borderRadius: 8, border: 'none',
          background: C.accent, color: '#fff', fontSize: 12.5, fontWeight: 600,
          cursor: available && !loading ? 'pointer' : 'not-allowed',
          opacity: available ? (loading ? 0.7 : 1) : 0.5,
        }}>
          {loading ? 'Running conversation…' : '▶ Run agentic pipeline'}
        </button>
      </div>

      {result && (
        <>
          {result.fallback_used && (
            <div style={{ marginBottom: 12, padding: '10px 14px', background: C.card,
              border: `1px solid ${C.amber}`, borderRadius: 10, fontSize: 12, color: C.textSec }}>
              Fell back to the deterministic pipeline: {result.fallback_reason}
            </div>
          )}

          {!result.fallback_used && (
            <div style={{ marginBottom: 12, fontSize: 11.5, color: C.textMut }}>
              {result.rounds_used} Critic-triggered re-run{result.rounds_used === 1 ? '' : 's'} ·{' '}
              {result.agentic_trace?.length || 0} messages
            </div>
          )}

          {result.agentic_trace?.length > 0 && (
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
              padding: '4px 16px', marginBottom: 16 }}>
              {result.agentic_trace.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
            </div>
          )}

          {result.metrics && (
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 18 }}>
              <SectionHeader>Final Metrics</SectionHeader>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10 }}>
                {[
                  ['FDR@25%', `${(result.metrics.fdr_25 * 100).toFixed(1)}%`],
                  ['FDR@50%', `${(result.metrics.fdr_50 * 100).toFixed(1)}%`],
                  ['FDR@100%', `${(result.metrics.fdr_100 * 100).toFixed(1)}%`],
                  ['TTFF', `#${result.metrics.ttff}`],
                  ['Tests run', result.metrics.tests_run],
                ].map(([label, value]) => (
                  <div key={label} style={{ background: C.surface, borderRadius: 8, padding: '9px 12px' }}>
                    <div style={{ fontSize: 10, color: C.textMut, marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: C.textPri }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
