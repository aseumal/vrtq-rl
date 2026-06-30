import SectionHeader from './SectionHeader'
import { C } from './constants'

const ARROW = '→'

function Box({ title, sub, tag, tagColor, tagBg, width }) {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`, borderRadius: 10,
      padding: '12px 14px', minWidth: width || 150, flex: '1 1 0',
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: C.textPri, marginBottom: 4 }}>
        {title}
      </div>
      {sub && (
        <div style={{ fontSize: 10.5, color: C.textSec, lineHeight: 1.5, marginBottom: tag ? 8 : 0 }}>
          {sub}
        </div>
      )}
      {tag && (
        <span style={{
          display: 'inline-block', background: tagBg || C.accentLo, color: tagColor || C.accent,
          fontSize: 9.5, padding: '2px 7px', borderRadius: 99, fontWeight: 500,
        }}>{tag}</span>
      )}
    </div>
  )
}

function Connector() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: C.textMut, fontSize: 16, padding: '0 4px', flex: '0 0 auto',
    }}>{ARROW}</div>
  )
}

function Row({ boxes }) {
  return (
    <div style={{ display: 'flex', alignItems: 'stretch', flexWrap: 'wrap', gap: 4 }}>
      {boxes.map((b, i) => (
        <div key={b.title} style={{ display: 'flex', alignItems: 'center', flex: '1 1 0', minWidth: 150 }}>
          <Box {...b} />
          {i < boxes.length - 1 && <Connector />}
        </div>
      ))}
    </div>
  )
}

function Layer({ label, children }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 18 }}>
      <div style={{ fontSize: 10, color: C.textMut, textTransform: 'uppercase',
        letterSpacing: '0.08em', marginBottom: 12, fontWeight: 600 }}>{label}</div>
      {children}
    </div>
  )
}

export default function ArchitectureTab() {
  return (
    <div>
      <SectionHeader>System Architecture</SectionHeader>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 16 }}>
        <Layer label="1. Offline Training">
          <Row boxes={[
            { title: 'Data Generation', sub: 'synthetic_test_suite.py + fault_injection.py — 200 synthetic tests, risk-biased fault injection' },
            { title: 'Gymnasium Env', sub: 'test_prioritization_env.py — 2000-dim obs, Discrete(200) masked actions' },
            { title: 'SB3 Training', sub: 'MaskablePPO + DQN, multi-seed train/eval split', tag: 'RL', tagColor: C.teal, tagBg: C.tealBg },
            { title: 'MLflow Tracking', sub: 'evaluation/mlflow_logger.py — params, metrics, run history' },
            { title: 'Saved Model', sub: 'models/*.zip — loaded by TestSelectorAgent at inference time' },
          ]} />
        </Layer>

        <Layer label="2. Online Agent Pipeline — agents/orchestrator.py (sequential, not AutoGen GroupChat)">
          <Row boxes={[
            {
              title: 'ChangeAnalyzerAgent', sub: 'Regex-parses git diff, extracts modules/churn/depth',
              tag: 'Rule-based + optional LLM (gated off)', tagColor: C.purple, tagBg: C.purpleBg,
            },
            {
              title: 'RiskScorerAgent', sub: 'Builds VRTQ state matrix deterministically',
              tag: 'Deterministic — no LLM', tagColor: C.textSec, tagBg: C.surface,
            },
            {
              title: 'TestSelectorAgent', sub: 'Runs MaskablePPO inference (or VRTQ fallback)',
              tag: 'RL inference — no LLM', tagColor: C.teal, tagBg: C.tealBg,
            },
            {
              title: 'ReportAgent', sub: 'Computes FDR/TTFF/TSR, formats top tests',
              tag: 'Deterministic + optional LLM (gated off)', tagColor: C.purple, tagBg: C.purpleBg,
            },
          ]} />
        </Layer>

        <Layer label="3. Serving">
          <Row boxes={[
            { title: 'FastAPI', sub: 'api/main.py — /api/prioritize, /api/compare, /api/learning-curve' },
            { title: 'React Dashboard', sub: 'This app — Vite + Recharts, polls the API above' },
          ]} />
        </Layer>

        <Layer label="4. Agentic Mode — agents/agentic_orchestrator.py (opt-in, requires OPENAI_API_KEY)">
          <Row boxes={[
            {
              title: 'Supervisor', sub: 'AssistantAgent — calls the same 4 stages above as real tools, in order',
              tag: 'Tool-calling LLM', tagColor: C.purple, tagBg: C.purpleBg,
            },
            {
              title: 'ToolExecutor', sub: 'UserProxyAgent — actually executes the tool calls Supervisor requests',
              tag: 'Function execution', tagColor: C.textSec, tagBg: C.surface,
            },
            {
              title: 'Critic', sub: 'AssistantAgent — reviews metrics, can REQUEST a re-run with adjusted budget/focus',
              tag: 'Autonomous review', tagColor: C.purple, tagBg: C.purpleBg,
            },
          ]} />
        </Layer>
      </div>

      <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 }}>
        <SectionHeader>Is this "agentic AI"? Was AutoGen really used?</SectionHeader>
        <p style={{ color: C.textSec, lineHeight: 1.7, fontSize: 12.5, margin: '0 0 12px' }}>
          The default pipeline (Layer 2) stays exactly what it always was: a{' '}
          <strong style={{ color: C.textPri }}>deterministic 4-stage Python pipeline</strong>, not a multi-agent
          conversation. <code style={{ color: C.teal }}>RiskScorerAgent</code> and{' '}
          <code style={{ color: C.teal }}>TestSelectorAgent</code> — the stages that actually decide which tests to
          run — are pure deterministic/RL Python with zero AutoGen involvement, and this is what every evaluation
          script (<code>compare_baselines.py</code>, <code>validate_model.py</code>, <code>run_seed_sweep.py</code>)
          exercises. It is never invoked by the agentic mode below.
        </p>
        <p style={{ color: C.textSec, lineHeight: 1.7, fontSize: 12.5, margin: 0 }}>
          A separate, <strong style={{ color: C.textPri }}>opt-in</strong> agentic mode (Layer 4, see the{' '}
          <strong style={{ color: C.accent }}>Agentic</strong> tab) adds genuine multi-turn AutoGen collaboration:
          a Supervisor agent drives the same four pipeline
          stages via real tool/function calls, and a Critic agent autonomously reviews the resulting metrics —
          if FDR@50% is too low or selection is too concentrated in one module, the Critic can issue a{' '}
          <code>REQUEST:</code> that triggers a genuine re-run with adjusted parameters, not just more chat text.
          This requires a real <code>OPENAI_API_KEY</code> and gracefully falls back to the deterministic pipeline
          above if one isn't configured.
        </p>
      </div>
    </div>
  )
}
