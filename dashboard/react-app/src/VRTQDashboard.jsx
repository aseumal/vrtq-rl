import { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart,
  PolarGrid, PolarAngleAxis, Radar
} from "recharts";

// ─── Sample data (replace with API call to FastAPI backend) ──────────────────
const SAMPLE_DATA = {
  report: {
    method: "VRTQ Heuristic (fallback — train PPO to activate RL)",
    metrics: {
      fdr_25: 0.0811, fdr_50: 0.1622, fdr_100: 0.4054,
      ttff: 3, tsr: 0.75, faults_found: 15, total_faults: 37,
      tests_run: 50, total_tests: 200, estimated_time_saved_seconds: 2180.9,
    },
    top_tests: [
      { rank: 1, test_id: "TEST_155", module: "payment_service", test_type: "integration", vrtq_composite: 0.9, rationale: "in changed module, high risk (0.92), failure rate 33%", execution_time_seconds: 18.2 },
      { rank: 2, test_id: "TEST_174", module: "auth_service",    test_type: "integration", vrtq_composite: 0.851, rationale: "in changed module, high risk (0.98), failure rate 35%", execution_time_seconds: 22.1 },
      { rank: 3, test_id: "TEST_110", module: "auth_service",    test_type: "e2e",         vrtq_composite: 0.813, rationale: "in changed module, high risk (1.00), failure rate 53%", execution_time_seconds: 61.4 },
      { rank: 4, test_id: "TEST_177", module: "reporting_engine",test_type: "integration", vrtq_composite: 0.794, rationale: "high risk (1.00), failure rate 37%, not run in 30 days", execution_time_seconds: 14.8 },
      { rank: 5, test_id: "TEST_141", module: "payment_service", test_type: "unit",        vrtq_composite: 0.782, rationale: "in changed module, high risk (1.00), failure rate 46%", execution_time_seconds: 3.2 },
      { rank: 6, test_id: "TEST_089", module: "auth_service",    test_type: "unit",        vrtq_composite: 0.761, rationale: "in changed module, high risk (0.88)", execution_time_seconds: 4.1 },
      { rank: 7, test_id: "TEST_032", module: "payment_service", test_type: "e2e",         vrtq_composite: 0.743, rationale: "in changed module, failure rate 28%", execution_time_seconds: 72.3 },
      { rank: 8, test_id: "TEST_198", module: "order_processing",test_type: "integration", vrtq_composite: 0.721, rationale: "high risk (0.85), not run in 14 days", execution_time_seconds: 19.7 },
      { rank: 9, test_id: "TEST_067", module: "api_gateway",     test_type: "integration", vrtq_composite: 0.708, rationale: "high risk (0.79)", execution_time_seconds: 11.3 },
      { rank: 10,test_id: "TEST_023", module: "payment_service", test_type: "unit",        vrtq_composite: 0.691, rationale: "in changed module, failure rate 21%", execution_time_seconds: 2.8 },
    ],
    risk_summary: { modules_affected: ["auth_service","payment_service"], n_affected_tests: 57, avg_risk_score: 0.727, high_risk_tests: 33, churn_score: 0.96, dependency_depth: 4 },
    git_context: { modules_affected: ["auth_service","payment_service"], change_type: "feature", churn_score: 0.96 },
    summary: "This feature affects auth_service and payment_service — two of the highest-criticality modules. VRTQ-RL selected 50 tests prioritising high failure-rate and recently-changed paths, detecting 16% of faults at the 50% mark with first failure at test #3. Running the full suite would take ~36 min more.",
  },
  comparison: [
    { method: "Random",         fdr_25: 0.243, fdr_50: 0.486, fdr_100: 0.243, ttff: 11, tsr: 0.05 },
    { method: "VRTQ Heuristic", fdr_25: 0.081, fdr_50: 0.162, fdr_100: 0.405, ttff: 3,  tsr: 0.75 },
    { method: "DQN",            fdr_25: 0.0,   fdr_50: 0.0,   fdr_100: 0.0,   ttff: 0,  tsr: 0.0  },
    { method: "PPO (VRTQ-RL)",  fdr_25: 0.0,   fdr_50: 0.0,   fdr_100: 0.0,   ttff: 0,  tsr: 0.0  },
  ],
};

// ─── Learning curve (simulated — replace with MLflow data) ───────────────────
const LEARNING_CURVE = Array.from({ length: 20 }, (_, i) => ({
  step: (i + 1) * 5000,
  ppo: Math.min(0.72, 0.15 + 0.57 * (1 - Math.exp(-i / 6)) + (Math.random() * 0.03 - 0.015)),
  vrtq: 0.405,
  random: 0.243,
}));

// ─── Palette ─────────────────────────────────────────────────────────────────
const C = {
  bg:       "#0b0f1a",
  surface:  "#111827",
  card:     "#1a2236",
  border:   "#1e2d47",
  accent:   "#3b82f6",
  accentLo: "#1d3a6e",
  teal:     "#14b8a6",
  amber:    "#f59e0b",
  red:      "#ef4444",
  green:    "#22c55e",
  purple:   "#a78bfa",
  textPri:  "#f1f5f9",
  textSec:  "#94a3b8",
  textMut:  "#475569",
};

const METHOD_COLOR = {
  "Random":         C.textMut,
  "VRTQ Heuristic": C.amber,
  "DQN":            C.purple,
  "PPO (VRTQ-RL)":  C.teal,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const badge = (type) => {
  const map = { unit: [C.accentLo, C.accent], integration: ["#1a3a2a", C.teal], e2e: ["#2d1a3a", C.purple] };
  const [bg, color] = map[type] || [C.card, C.textSec];
  return (
    <span style={{ background: bg, color, fontSize: 11, padding: "2px 8px",
      borderRadius: 99, fontWeight: 500, fontFamily: "monospace" }}>
      {type}
    </span>
  );
};

const ModulePill = ({ name }) => {
  const isAffected = SAMPLE_DATA.report.risk_summary.modules_affected.includes(name);
  return (
    <span style={{ background: isAffected ? C.accentLo : C.card,
      color: isAffected ? C.accent : C.textSec,
      border: `1px solid ${isAffected ? C.accent : C.border}`,
      fontSize: 11, padding: "2px 8px", borderRadius: 99 }}>
      {name}
    </span>
  );
};

// ─── Sub-components ───────────────────────────────────────────────────────────
function MetricCard({ label, value, sub, accent, large }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: "16px 20px", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, left: 0, width: 3,
        height: "100%", background: accent || C.accent, borderRadius: "12px 0 0 12px" }} />
      <div style={{ fontSize: 11, color: C.textMut, textTransform: "uppercase",
        letterSpacing: "0.08em", marginBottom: 6, paddingLeft: 4 }}>{label}</div>
      <div style={{ fontSize: large ? 32 : 26, fontWeight: 700,
        color: accent || C.textPri, paddingLeft: 4, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: C.textSec, marginTop: 4, paddingLeft: 4 }}>{sub}</div>}
    </div>
  );
}

function SectionHeader({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
      <div style={{ width: 3, height: 18, background: C.accent, borderRadius: 2 }} />
      <span style={{ fontSize: 13, fontWeight: 600, color: C.textSec,
        textTransform: "uppercase", letterSpacing: "0.08em" }}>{children}</span>
    </div>
  );
}

function TestRow({ test, index }) {
  const [open, setOpen] = useState(false);
  const score = test.vrtq_composite;
  const barW = `${(score * 100).toFixed(0)}%`;

  return (
    <div style={{ borderBottom: `1px solid ${C.border}`, transition: "background 0.15s",
      background: open ? C.card : "transparent" }}>
      <div onClick={() => setOpen(!open)}
        style={{ display: "grid", gridTemplateColumns: "32px 90px 1fr 120px 80px 60px",
          gap: 12, padding: "10px 16px", cursor: "pointer", alignItems: "center",
          fontSize: 13 }}>
        <span style={{ color: C.textMut, fontWeight: 600 }}>#{test.rank}</span>
        <span style={{ color: C.accent, fontFamily: "monospace", fontSize: 12 }}>{test.test_id}</span>
        <ModulePill name={test.module} />
        {badge(test.test_type)}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ flex: 1, height: 4, background: C.border, borderRadius: 2 }}>
            <div style={{ width: barW, height: "100%",
              background: score > 0.8 ? C.red : score > 0.6 ? C.amber : C.teal,
              borderRadius: 2, transition: "width 0.5s" }} />
          </div>
          <span style={{ color: C.textSec, fontSize: 11, minWidth: 32 }}>{score.toFixed(3)}</span>
        </div>
        <span style={{ color: open ? C.accent : C.textMut, textAlign: "right" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{ padding: "0 16px 14px 52px", fontSize: 12, color: C.textSec, lineHeight: 1.7 }}>
          <div><span style={{ color: C.textMut }}>Rationale: </span>{test.rationale}</div>
          <div><span style={{ color: C.textMut }}>Exec time: </span>{test.execution_time_seconds}s</div>
          <div style={{ display: "flex", gap: 16, marginTop: 6 }}>
            <span>V={((score - 0.35*0.7 - 0.20*0.6 - 0.15*0.5)/0.30).toFixed(2)}</span>
            <span>R={(test.vrtq_composite > 0.8 ? 0.92 : 0.71).toFixed(2)}</span>
            <span>T={((1 - test.execution_time_seconds/120)).toFixed(2)}</span>
            <span>Q=0.50</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ComparisonChart({ data }) {
  const formatted = data.map(d => ({
    ...d,
    "FDR@25%": +(d.fdr_25 * 100).toFixed(1),
    "FDR@50%": +(d.fdr_50 * 100).toFixed(1),
    "FDR@100%": +(d.fdr_100 * 100).toFixed(1),
    "TSR %": +(d.tsr * 100).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={formatted} barCategoryGap="25%" barGap={3}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
        <XAxis dataKey="method" tick={{ fill: C.textSec, fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: C.textMut, fontSize: 10 }} axisLine={false} tickLine={false}
          tickFormatter={v => `${v}%`} domain={[0, 100]} />
        <Tooltip contentStyle={{ background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 8, color: C.textPri }} formatter={(v) => [`${v}%`]} />
        <Legend wrapperStyle={{ fontSize: 11, color: C.textSec }} />
        <Bar dataKey="FDR@25%" fill={C.accent} radius={[3,3,0,0]} />
        <Bar dataKey="FDR@50%" fill={C.teal} radius={[3,3,0,0]} />
        <Bar dataKey="TSR %" fill={C.amber} radius={[3,3,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function LearningCurve({ data }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
        <XAxis dataKey="step" tick={{ fill: C.textMut, fontSize: 10 }}
          tickFormatter={v => `${v/1000}k`} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: C.textMut, fontSize: 10 }} axisLine={false} tickLine={false}
          tickFormatter={v => pct(v)} domain={[0, 0.9]} />
        <Tooltip contentStyle={{ background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 8, color: C.textPri }} formatter={v => [pct(v), "FDR"]} />
        <Line dataKey="ppo" name="PPO (VRTQ-RL)" stroke={C.teal} strokeWidth={2}
          dot={false} strokeDasharray="0" />
        <Line dataKey="vrtq" name="VRTQ Heuristic" stroke={C.amber} strokeWidth={1.5}
          dot={false} strokeDasharray="5 3" />
        <Line dataKey="random" name="Random" stroke={C.textMut} strokeWidth={1}
          dot={false} strokeDasharray="3 3" />
        <Legend wrapperStyle={{ fontSize: 11, color: C.textSec }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function VRTQRadar({ test }) {
  const data = [
    { axis: "Value",   score: test ? 0.72 : 0 },
    { axis: "Risk",    score: test ? test.vrtq_composite * 1.05 : 0 },
    { axis: "Time",    score: test ? 1 - test.execution_time_seconds / 120 : 0 },
    { axis: "Quality", score: test ? 0.60 : 0 },
  ];
  return (
    <ResponsiveContainer width="100%" height={160}>
      <RadarChart data={data}>
        <PolarGrid stroke={C.border} />
        <PolarAngleAxis dataKey="axis" tick={{ fill: C.textSec, fontSize: 11 }} />
        <Radar dataKey="score" stroke={C.accent} fill={C.accent} fillOpacity={0.2}
          strokeWidth={2} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────
export default function VRTQDashboard() {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedTest, setSelectedTest] = useState(null);
  const [animIn, setAnimIn] = useState(false);

  const { report, comparison } = SAMPLE_DATA;
  const m = report.metrics;

  useEffect(() => {
    setTimeout(() => setAnimIn(true), 100);
  }, []);

  const tabs = [
    { id: "overview",  label: "Overview" },
    { id: "tests",     label: "Test Queue" },
    { id: "compare",   label: "Baselines" },
    { id: "training",  label: "Learning Curve" },
  ];

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.textPri,
      fontFamily: "'Inter', system-ui, sans-serif", fontSize: 14 }}>

      {/* Header */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: "0 32px",
        display: "flex", alignItems: "center", gap: 0, height: 56,
        background: C.surface, position: "sticky", top: 0, zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginRight: 40 }}>
          <div style={{ width: 28, height: 28, borderRadius: 6, background: C.accentLo,
            border: `1px solid ${C.accent}`, display: "flex", alignItems: "center",
            justifyContent: "center", fontSize: 13, fontWeight: 700, color: C.accent }}>
            RL
          </div>
          <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.02em" }}>VRTQ-RL</span>
          <span style={{ color: C.textMut, fontSize: 12 }}>Test Prioritization</span>
        </div>

        <div style={{ display: "flex", gap: 2 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              style={{ padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
                fontSize: 13, fontWeight: activeTab === t.id ? 600 : 400,
                background: activeTab === t.id ? C.accentLo : "transparent",
                color: activeTab === t.id ? C.accent : C.textSec,
                transition: "all 0.15s" }}>
              {t.label}
            </button>
          ))}
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.green,
            boxShadow: `0 0 6px ${C.green}` }} />
          <span style={{ fontSize: 12, color: C.textSec }}>Pipeline active</span>
          <div style={{ background: C.accentLo, color: C.accent, fontSize: 11,
            padding: "3px 10px", borderRadius: 99, fontWeight: 500 }}>
            {report.method.split("(")[0].trim()}
          </div>
        </div>
      </div>

      {/* Git context banner */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: "8px 32px", display: "flex", alignItems: "center", gap: 16, fontSize: 12 }}>
        <span style={{ color: C.textMut }}>Last diff:</span>
        {report.git_context.modules_affected.map(m => (
          <ModulePill key={m} name={m} />
        ))}
        <span style={{ color: C.textMut }}>·</span>
        <span style={{ background: C.card, color: C.textSec, padding: "2px 8px",
          borderRadius: 99, border: `1px solid ${C.border}` }}>
          {report.git_context.change_type}
        </span>
        <span style={{ color: C.textMut }}>·</span>
        <span style={{ color: C.textSec }}>churn {(report.git_context.churn_score * 100).toFixed(0)}%</span>
        <span style={{ color: C.textMut }}>·</span>
        <span style={{ color: C.textSec }}>{m.total_tests} tests · budget {m.tests_run}</span>
      </div>

      <div style={{ padding: "28px 32px", opacity: animIn ? 1 : 0,
        transform: animIn ? "none" : "translateY(8px)", transition: "all 0.3s" }}>

        {/* ── Overview ─────────────────────────────────────────────── */}
        {activeTab === "overview" && (
          <div>
            {/* Metric cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: 12, marginBottom: 28 }}>
              <MetricCard label="FDR @ 25%" value={pct(m.fdr_25)} accent={C.red}
                sub={`${Math.round(m.fdr_25 * m.total_faults)} faults early`} />
              <MetricCard label="FDR @ 50%" value={pct(m.fdr_50)} accent={C.amber}
                sub={`of ${m.total_faults} total faults`} />
              <MetricCard label="FDR @ 100%" value={pct(m.fdr_100)} accent={C.teal}
                sub={`${m.faults_found} found in budget`} />
              <MetricCard label="Time to first fault" value={`#${m.ttff}`} accent={C.green}
                sub="test position" />
              <MetricCard label="Suite reduction" value={pct(m.tsr)} accent={C.purple}
                sub={`${m.total_tests - m.tests_run} tests skipped`} large />
              <MetricCard label="Time saved" value={`${(m.estimated_time_saved_seconds/60).toFixed(0)}m`}
                accent={C.accent} sub="vs full suite run" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20 }}>
              {/* Summary */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`,
                borderRadius: 12, padding: 20 }}>
                <SectionHeader>AI Summary</SectionHeader>
                <p style={{ color: C.textSec, lineHeight: 1.8, margin: 0, fontSize: 13 }}>
                  {report.summary}
                </p>

                <div style={{ marginTop: 20, display: "grid",
                  gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {[
                    ["Affected modules", report.risk_summary.modules_affected.length],
                    ["Affected tests", report.risk_summary.n_affected_tests],
                    ["High-risk tests", report.risk_summary.high_risk_tests],
                    ["Avg risk score", report.risk_summary.avg_risk_score.toFixed(3)],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: C.surface, borderRadius: 8,
                      padding: "10px 14px", border: `1px solid ${C.border}` }}>
                      <div style={{ fontSize: 11, color: C.textMut, marginBottom: 2 }}>{k}</div>
                      <div style={{ fontSize: 18, fontWeight: 600, color: C.textPri }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* VRTQ radar */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`,
                borderRadius: 12, padding: 20 }}>
                <SectionHeader>VRTQ Score Profile</SectionHeader>
                <div style={{ fontSize: 12, color: C.textSec, marginBottom: 8 }}>
                  Top test: <span style={{ color: C.accent }}>
                    {report.top_tests[0]?.test_id}
                  </span>
                </div>
                <VRTQRadar test={report.top_tests[0]} />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 8 }}>
                  {[["Value", "0.30"], ["Risk", "0.35"], ["Time", "0.20"], ["Quality", "0.15"]].map(([k, w]) => (
                    <div key={k} style={{ fontSize: 11, color: C.textSec, display: "flex",
                      justifyContent: "space-between", padding: "2px 0" }}>
                      <span>{k}</span>
                      <span style={{ color: C.textMut }}>w={w}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Test Queue ────────────────────────────────────────────── */}
        {activeTab === "tests" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 16 }}>
              <SectionHeader>Prioritized Test Queue — top {report.top_tests.length} of {m.tests_run} selected</SectionHeader>
              <span style={{ fontSize: 12, color: C.textMut }}>Click any row to expand</span>
            </div>

            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
              overflow: "hidden" }}>
              {/* Table header */}
              <div style={{ display: "grid",
                gridTemplateColumns: "32px 90px 1fr 120px 80px 60px",
                gap: 12, padding: "8px 16px", borderBottom: `1px solid ${C.border}`,
                fontSize: 11, color: C.textMut, textTransform: "uppercase",
                letterSpacing: "0.06em" }}>
                <span>#</span><span>Test ID</span><span>Module</span>
                <span>Type</span><span>VRTQ</span><span></span>
              </div>

              {report.top_tests.map((t, i) => (
                <TestRow key={t.test_id} test={t} index={i} />
              ))}
            </div>

            <div style={{ marginTop: 12, fontSize: 12, color: C.textMut }}>
              Showing top 10 · {m.tests_run - 10} more tests selected but not shown ·
              Full list available via <code style={{ color: C.accent }}>python -m agents.orchestrator --save</code>
            </div>
          </div>
        )}

        {/* ── Baselines ─────────────────────────────────────────────── */}
        {activeTab === "compare" && (
          <div>
            <SectionHeader>Method Comparison</SectionHeader>
            <div style={{ background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 12, padding: 20, marginBottom: 20 }}>
              <ComparisonChart data={comparison} />
            </div>

            {/* Table */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 12, overflow: "hidden" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 90px 90px 90px 70px 70px",
                gap: 12, padding: "8px 16px", borderBottom: `1px solid ${C.border}`,
                fontSize: 11, color: C.textMut, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                <span>Method</span><span>FDR@25%</span><span>FDR@50%</span>
                <span>FDR@100%</span><span>TTFF</span><span>TSR</span>
              </div>
              {comparison.map((row) => (
                <div key={row.method} style={{ display: "grid",
                  gridTemplateColumns: "1fr 90px 90px 90px 70px 70px",
                  gap: 12, padding: "12px 16px", borderBottom: `1px solid ${C.border}`,
                  fontSize: 13 }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 10, height: 10, borderRadius: "50%",
                      background: METHOD_COLOR[row.method] || C.textMut, flexShrink: 0 }} />
                    <span style={{ color: METHOD_COLOR[row.method] || C.textSec,
                      fontWeight: row.method.includes("PPO") ? 600 : 400 }}>
                      {row.method}
                    </span>
                  </span>
                  {["fdr_25","fdr_50","fdr_100"].map(k => (
                    <span key={k} style={{ color: row[k] === 0 ? C.textMut : C.textPri }}>
                      {row[k] === 0 ? "—" : pct(row[k])}
                    </span>
                  ))}
                  <span style={{ color: row.ttff === 0 ? C.textMut : C.textPri }}>
                    {row.ttff === 0 ? "—" : `#${row.ttff}`}
                  </span>
                  <span style={{ color: row.tsr === 0 ? C.textMut : C.textPri }}>
                    {row.tsr === 0 ? "—" : pct(row.tsr)}
                  </span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 12, fontSize: 12, color: C.textMut }}>
              DQN and PPO show — until trained. Run <code style={{ color: C.accent }}>python -m rl.train_ppo</code> to populate.
            </div>
          </div>
        )}

        {/* ── Learning Curve ────────────────────────────────────────── */}
        {activeTab === "training" && (
          <div>
            <SectionHeader>PPO Learning Curve (simulated — replace with MLflow data)</SectionHeader>
            <div style={{ background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 12, padding: 20, marginBottom: 20 }}>
              <LearningCurve data={LEARNING_CURVE} />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
              {[
                { label: "Algorithm", value: "PPO", sub: "Proximal Policy Optimization" },
                { label: "Total timesteps", value: "100k", sub: "n_steps=2048, batch=64" },
                { label: "Discount factor γ", value: "0.99", sub: "patient long-horizon agent" },
                { label: "Learning rate", value: "3e-4", sub: "Adam optimizer" },
                { label: "Clip range ε", value: "0.2", sub: "policy update bound" },
                { label: "Entropy coef", value: "0.01", sub: "exploration bonus" },
              ].map(({ label, value, sub }) => (
                <div key={label} style={{ background: C.card, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: "14px 16px" }}>
                  <div style={{ fontSize: 11, color: C.textMut, marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 20, fontWeight: 600, color: C.accent }}>{value}</div>
                  <div style={{ fontSize: 11, color: C.textMut, marginTop: 2 }}>{sub}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
