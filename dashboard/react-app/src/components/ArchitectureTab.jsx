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
              title: 'Supervisor', sub: 'AssistantAgent — calls 10 tools across 5 phases as it decides, not a fixed script',
              tag: 'Tool-calling LLM', tagColor: C.purple, tagBg: C.purpleBg,
            },
            {
              title: 'ToolExecutor', sub: 'UserProxyAgent — actually executes the tool calls Supervisor requests',
              tag: 'Function execution', tagColor: C.textSec, tagBg: C.surface,
            },
            {
              title: 'RiskReviewer', sub: 'Phase 0, before scoring — can challenge an ambiguous module classification',
              tag: 'Pre-emptive challenge', tagColor: C.purple, tagBg: C.purpleBg,
            },
            {
              title: 'SLAReviewer', sub: 'Phase 2 — can force a time-budgeted re-selection if execution time exceeds a CI SLA',
              tag: 'Process review', tagColor: C.purple, tagBg: C.purpleBg,
            },
            {
              title: 'ConfidenceReviewer', sub: "Phase 2.5 — can swap a pick the RL policy was barely confident about",
              tag: 'Process review', tagColor: C.purple, tagBg: C.purpleBg,
            },
            {
              title: 'Critic', sub: 'Phase 3 — reviews final outcome quality; budget-increase requests must now be justified against the SLA ceiling',
              tag: 'Constrained outcome review', tagColor: C.purple, tagBg: C.purpleBg,
            },
          ]} />
        </Layer>
      </div>

      <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 }}>
        <SectionHeader>Is this "agentic AI"? Was AutoGen really used?</SectionHeader>
        <p style={{ color: C.textSec, lineHeight: 1.7, fontSize: 12.5, margin: '0 0 12px' }}>
          The default pipeline (Layer 2) stays exactly what it always was: a{' '}
          <strong style={{ color: C.textPri }}>deterministic 4-stage Python pipeline</strong>, not a multi-agent
          conversation. The methods that run the actual selection algorithm —{' '}
          <code style={{ color: C.teal }}>RiskScorerAgent.score()</code>'s VRTQ state-matrix math and{' '}
          <code style={{ color: C.teal }}>TestSelectorAgent.select()</code>'s PPO inference — are pure
          deterministic/RL Python; neither file has ever been modified by the agentic mode below and neither
          imports AutoGen. This is what every evaluation script (<code>compare_baselines.py</code>,{' '}
          <code>validate_model.py</code>, <code>run_seed_sweep.py</code>) exercises, untouched.
        </p>
        <p style={{ color: C.textSec, lineHeight: 1.7, fontSize: 12.5, margin: '0 0 12px' }}>
          That's a narrower claim than "AutoGen never affects which tests run," though — it doesn't, while you're
          on the Layer 2 default path. Once you opt into Layer 4, that's no longer true: the{' '}
          <em>algorithm</em> (the neural net forward pass, the VRTQ formula) is still never touched by an LLM, but
          the <em>inputs</em> to it genuinely are — RiskReviewer's module correction changes what
          <code style={{ color: C.teal }}> score_risk()</code> computes, SLAReviewer's time-budget request and
          ConfidenceReviewer's substitution directly change which test IDs end up in the final selection, and the
          Critic's budget/focus decisions do too. So in agentic mode, AutoGen has real causal influence over the
          outcome — it just exercises that influence by choosing arguments and corrections for still-deterministic
          functions to act on, not by replacing the functions themselves.
        </p>
        <p style={{ color: C.textSec, lineHeight: 1.7, fontSize: 12.5, margin: '0 0 12px' }}>
          A separate, <strong style={{ color: C.textPri }}>opt-in</strong> agentic mode (Layer 4, see the{' '}
          <strong style={{ color: C.accent }}>Agentic</strong> tab) adds genuine multi-turn AutoGen collaboration —
          and this is the part that actually earns the label: four different agents can each catch something the
          others got wrong, and each can change the final outcome, not just generate more chat text.{' '}
          <code style={{ color: C.purple }}>RiskReviewer</code> challenges an ambiguous module classification{' '}
          <em>before</em> risk scoring ever runs on it (state-matrix dependency — fixing it after the fact would
          need a full re-score). <code style={{ color: C.purple }}>SLAReviewer</code> checks the selection's real
          execution time against a CI budget and can force a genuinely different, time-truncated re-selection.{' '}
          <code style={{ color: C.purple }}>ConfidenceReviewer</code> inspects the PPO policy's actual per-pick
          action probabilities and can substitute out the one pick the model was least sure about. The{' '}
          <code style={{ color: C.purple }}>Critic</code> reviews final quality — but is now SLA-aware: it must
          compute the projected time cost of any budget increase and is structurally blocked from requesting one
          that would breach the ceiling, forcing it to justify a tradeoff explicitly instead of just asking for
          more. Every reviewer's round count is capped by a hard Python counter, independent of what the LLM says.
        </p>
        <p style={{ color: C.textSec, lineHeight: 1.7, fontSize: 12.5, margin: 0 }}>
          This requires a real <code>OPENAI_API_KEY</code> and gracefully falls back to the deterministic pipeline
          above if one isn't configured — verified to produce byte-identical output to the plain pipeline in that
          case. It's also fully separate from the RL evaluation scripts above: none of them import anything from
          this layer, so nothing here can affect the validated PPO results.
        </p>
      </div>
    </div>
  )
}
