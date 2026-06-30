/**
 * App.jsx — VRTQ-RL React Dashboard
 * Main entry point. Manages global state and tab routing.
 *
 * Author: Anthony Vallente
 * Project: VRTQ-RL
 */
import { useState, useEffect, useCallback } from 'react'
import OverviewTab   from './components/OverviewTab'
import TestQueueTab  from './components/TestQueueTab'
import BaselinesTab  from './components/BaselinesTab'
import TrainingTab   from './components/TrainingTab'
import ArchitectureTab from './components/ArchitectureTab'
import AgentChatTab   from './components/AgentChatTab'
import ModulePill    from './components/ModulePill'
import { C, MODULES, pct } from './components/constants'

// ── Fallback data (used when API is unreachable) ──────────────────────────────
const FALLBACK_REPORT = {
  method: 'VRTQ Heuristic (fallback)',
  metrics: { fdr_25:0.081, fdr_50:0.162, fdr_100:0.405, ttff:3, tsr:0.75,
    faults_found:15, total_faults:37, tests_run:50, total_tests:200,
    estimated_time_saved_seconds:2180 },
  top_tests: [
    {rank:1,test_id:'TEST_155',module:'payment_service', test_type:'integration',vrtq_composite:0.900,rationale:'in changed module, high risk (0.92)',execution_time_seconds:18.2},
    {rank:2,test_id:'TEST_174',module:'auth_service',    test_type:'integration',vrtq_composite:0.851,rationale:'in changed module, high risk (0.98)',execution_time_seconds:22.1},
    {rank:3,test_id:'TEST_110',module:'auth_service',    test_type:'e2e',        vrtq_composite:0.813,rationale:'high risk (1.00), failure rate 53%', execution_time_seconds:61.4},
    {rank:4,test_id:'TEST_177',module:'reporting_engine',test_type:'integration',vrtq_composite:0.794,rationale:'high risk, not run in 30 days',       execution_time_seconds:14.8},
    {rank:5,test_id:'TEST_141',module:'payment_service', test_type:'unit',       vrtq_composite:0.782,rationale:'high risk (1.00), failure rate 46%', execution_time_seconds:3.2},
    {rank:6,test_id:'TEST_089',module:'auth_service',    test_type:'unit',       vrtq_composite:0.761,rationale:'in changed module, high risk (0.88)',execution_time_seconds:4.1},
    {rank:7,test_id:'TEST_032',module:'payment_service', test_type:'e2e',        vrtq_composite:0.743,rationale:'failure rate 28%',                   execution_time_seconds:72.3},
    {rank:8,test_id:'TEST_198',module:'order_processing',test_type:'integration',vrtq_composite:0.721,rationale:'high risk (0.85), stale',             execution_time_seconds:19.7},
    {rank:9,test_id:'TEST_067',module:'api_gateway',     test_type:'integration',vrtq_composite:0.708,rationale:'high risk (0.79)',                    execution_time_seconds:11.3},
    {rank:10,test_id:'TEST_023',module:'payment_service',test_type:'unit',       vrtq_composite:0.691,rationale:'failure rate 21%',                   execution_time_seconds:2.8},
  ],
  risk_summary:{modules_affected:['auth_service','payment_service'],n_affected_tests:57,avg_risk_score:0.727,high_risk_tests:33,churn_score:0.96,dependency_depth:4},
  git_context:{modules_affected:['auth_service','payment_service'],change_type:'feature',churn_score:0.96},
  summary:'This feature affects auth_service and payment_service — two of the highest-criticality modules. VRTQ-RL selected 50 tests prioritising high failure-rate and recently-changed paths, detecting 16% of faults at the 50% mark with first failure at test #3.',
}

const FALLBACK_COMPARISON = [
  {method:'Random',        fdr_25:0.243,fdr_50:0.486,fdr_100:0.243,ttff:11,tsr:0.05},
  {method:'VRTQ Heuristic',fdr_25:0.081,fdr_50:0.162,fdr_100:0.405,ttff:3, tsr:0.75},
  {method:'DQN',           fdr_25:0,    fdr_50:0,    fdr_100:0,    ttff:0, tsr:0},
  {method:'PPO (VRTQ-RL)', fdr_25:0,    fdr_50:0,    fdr_100:0,    ttff:0, tsr:0},
]

import { useMemo } from 'react'
const FALLBACK_CURVE = Array.from({length:20},(_,i)=>({
  step:(i+1)*5000,
  ppo:+Math.min(0.72,0.15+0.57*(1-Math.exp(-i/6))).toFixed(3),
  vrtq:0.405,random:0.243,
}))

// ── Tab definitions ───────────────────────────────────────────────────────────
const TABS = [
  {id:'overview', label:'Overview'},
  {id:'tests',    label:'Test Queue'},
  {id:'compare',  label:'Baselines'},
  {id:'training', label:'Learning Curve'},
  {id:'architecture', label:'Architecture'},
  {id:'agentic', label:'Agentic'},
]

const TECH_STACK = {
  Frontend: ['React 18', 'Vite', 'Recharts'],
  Backend:  ['FastAPI', 'Pydantic', 'AutoGen (4-agent pipeline)'],
  ML:       ['Stable-Baselines3 (PPO + DQN)', 'Gymnasium', 'pandas / numpy', 'MLflow'],
}

export default function App() {
  const [tab,        setTab]        = useState('overview')
  const [report,     setReport]     = useState(FALLBACK_REPORT)
  const [comparison, setComparison] = useState(FALLBACK_COMPARISON)
  const [curve,      setCurve]      = useState({data: FALLBACK_CURVE, source:'simulated'})
  const [status,     setStatus]     = useState(null)
  const [loading,    setLoading]    = useState({report:false,compare:false,curve:false,agentic:false})
  const [apiOnline,  setApiOnline]  = useState(false)
  const [animIn,     setAnimIn]     = useState(false)

  // Agentic mode
  const [agenticStatus, setAgenticStatus] = useState(null)
  const [agenticResult, setAgenticResult] = useState(null)

  // Diff controls
  const [selModules, setSelModules] = useState(['payment_service','auth_service'])
  const [churn,      setChurn]      = useState('medium')
  const [budget,     setBudget]     = useState(50)
  const [showConfig, setShowConfig] = useState(false)

  // ── Boot ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    setTimeout(() => setAnimIn(true), 80)
    checkApi()
  }, [])

  const checkApi = async () => {
    try {
      const r = await fetch('/api/health', {signal: AbortSignal.timeout(2000)})
      if (r.ok) {
        setApiOnline(true)
        const s = await fetch('/api/status').then(r=>r.json()).catch(()=>null)
        setStatus(s)
        runPrioritize()
        runCompare()
        runCurve()
        const a = await fetch('/api/agentic-status').then(r=>r.json()).catch(()=>null)
        setAgenticStatus(a)
      }
    } catch { setApiOnline(false) }
  }

  // ── API calls ─────────────────────────────────────────────────────────────
  const runPrioritize = useCallback(async () => {
    setLoading(l=>({...l,report:true}))
    try {
      const r = await fetch('/api/prioritize',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({modules:selModules,churn,budget,use_llm:false})})
      if (r.ok) setReport(await r.json())
    } catch {}
    setLoading(l=>({...l,report:false}))
  }, [selModules, churn, budget])

  const runCompare = useCallback(async () => {
    setLoading(l=>({...l,compare:true}))
    try {
      const r = await fetch(`/api/compare?budget=${budget}`)
      if (r.ok) { const d = await r.json(); setComparison(d.comparison) }
    } catch {}
    setLoading(l=>({...l,compare:false}))
  }, [budget])

  const runCurve = useCallback(async () => {
    setLoading(l=>({...l,curve:true}))
    try {
      const r = await fetch('/api/learning-curve')
      if (r.ok) setCurve(await r.json())
    } catch {}
    setLoading(l=>({...l,curve:false}))
  }, [])

  const runAgentic = useCallback(async () => {
    setLoading(l=>({...l,agentic:true}))
    try {
      const r = await fetch('/api/prioritize-agentic',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({modules:selModules,churn,budget})})
      if (r.ok) setAgenticResult(await r.json())
    } catch {}
    setLoading(l=>({...l,agentic:false}))
  }, [selModules, churn, budget])

  const toggleModule = (mod) => {
    setSelModules(prev =>
      prev.includes(mod)
        ? prev.length > 1 ? prev.filter(m => m !== mod) : prev
        : [...prev, mod]
    )
  }

  // ── Render ────────────────────────────────────────────────────────────────
  const m = report.metrics
  return (
    <div style={{background:C.bg,minHeight:'100vh',color:C.textPri,
      fontFamily:"'Inter',system-ui,sans-serif",fontSize:13}}>

      {/* ── Header ── */}
      <div style={{borderBottom:`1px solid ${C.border}`,padding:'0 24px',
        display:'flex',alignItems:'center',height:52,background:C.surface,
        position:'sticky',top:0,zIndex:10,gap:0}}>

        <div style={{display:'flex',alignItems:'center',gap:8,marginRight:32}}>
          <div style={{width:26,height:26,borderRadius:6,background:C.accentLo,
            border:`1px solid ${C.accent}`,display:'flex',alignItems:'center',
            justifyContent:'center',fontSize:11,fontWeight:700,color:C.accent}}>RL</div>
          <span style={{fontWeight:700,fontSize:14,letterSpacing:'-0.02em'}}>VRTQ-RL</span>
          <span style={{color:C.textMut,fontSize:11}}>Test Prioritization</span>
        </div>

        <div style={{display:'flex',gap:2}}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              padding:'5px 12px',borderRadius:6,border:'none',cursor:'pointer',fontSize:12,
              fontWeight:tab===t.id?600:400,
              background:tab===t.id?C.accentLo:'transparent',
              color:tab===t.id?C.accent:C.textSec,transition:'all 0.15s',
            }}>{t.label}</button>
          ))}
        </div>

        <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:10}}>
          {status && (
            <span style={{fontSize:11,color:C.textMut}}>
              {status.dataset.n_tests} tests · {status.dataset.n_faults} faults
            </span>
          )}
          <div style={{width:7,height:7,borderRadius:'50%',
            background: apiOnline ? C.green : C.amber}} />
          <span style={{fontSize:11,color:C.textSec}}>
            {apiOnline ? 'API connected' : 'offline (fallback data)'}
          </span>
          {status?.ppo_model_ready && (
            <div style={{background:C.tealBg,color:C.teal,fontSize:10,
              padding:'2px 8px',borderRadius:99,fontWeight:500}}>PPO ready</div>
          )}
        </div>
      </div>

      {/* ── Git context banner ── */}
      <div style={{background:C.surface,borderBottom:`1px solid ${C.border}`,
        padding:'0 24px',display:'flex',alignItems:'center',gap:0,minHeight:40}}>
        <div style={{display:'flex',alignItems:'center',gap:10,flex:1,flexWrap:'wrap',padding:'8px 0'}}>
          <span style={{fontSize:11,color:C.textMut}}>diff:</span>
          {report.git_context.modules_affected.map(mod=>(
            <ModulePill key={mod} name={mod} highlighted />
          ))}
          <span style={{color:C.textMut,fontSize:11}}>·</span>
          <span style={{background:C.card,color:C.textSec,padding:'2px 8px',
            borderRadius:99,border:`1px solid ${C.border}`,fontSize:11}}>
            {report.git_context.change_type}
          </span>
          <span style={{color:C.textMut,fontSize:11}}>·</span>
          <span style={{fontSize:11,color:C.textSec}}>
            churn {(report.git_context.churn_score*100).toFixed(0)}%
          </span>
          <span style={{color:C.textMut,fontSize:11}}>·</span>
          <span style={{fontSize:11,color:C.textSec}}>
            {m.total_tests} tests · budget {m.tests_run}
          </span>
        </div>

        <button onClick={()=>setShowConfig(!showConfig)} style={{
          padding:'4px 12px',borderRadius:6,border:`1px solid ${C.border}`,
          background:showConfig?C.accentLo:C.card,color:showConfig?C.accent:C.textSec,
          fontSize:11,cursor:'pointer',whiteSpace:'nowrap',marginLeft:12,
        }}>
          {showConfig ? '✕ close' : '⚙ configure diff'}
        </button>
      </div>

      {/* ── Diff config panel ── */}
      {showConfig && (
        <div style={{background:C.surface,borderBottom:`1px solid ${C.border}`,
          padding:'16px 24px',display:'flex',alignItems:'flex-start',gap:24,flexWrap:'wrap'}}>

          <div>
            <div style={{fontSize:11,color:C.textMut,marginBottom:8,textTransform:'uppercase',letterSpacing:'0.06em'}}>
              Changed modules
            </div>
            <div style={{display:'flex',flexWrap:'wrap',gap:6}}>
              {MODULES.map(mod=>(
                <button key={mod} onClick={()=>toggleModule(mod)} style={{
                  padding:'3px 10px',borderRadius:99,fontSize:11,cursor:'pointer',border:'none',
                  background:selModules.includes(mod)?C.accentLo:C.card,
                  color:selModules.includes(mod)?C.accent:C.textSec,
                  border:`1px solid ${selModules.includes(mod)?C.accent:C.border}`,
                }}>{mod}</button>
              ))}
            </div>
          </div>

          <div>
            <div style={{fontSize:11,color:C.textMut,marginBottom:8,textTransform:'uppercase',letterSpacing:'0.06em'}}>Churn</div>
            <div style={{display:'flex',gap:6}}>
              {['low','medium','high'].map(c=>(
                <button key={c} onClick={()=>setChurn(c)} style={{
                  padding:'4px 12px',borderRadius:6,border:'none',cursor:'pointer',fontSize:12,
                  background:churn===c?C.accentLo:C.card,
                  color:churn===c?C.accent:C.textSec,
                  border:`1px solid ${churn===c?C.accent:C.border}`,
                }}>{c}</button>
              ))}
            </div>
          </div>

          <div>
            <div style={{fontSize:11,color:C.textMut,marginBottom:8,textTransform:'uppercase',letterSpacing:'0.06em'}}>
              Budget: {budget} tests
            </div>
            <input type="range" min={10} max={100} step={5} value={budget}
              onChange={e=>setBudget(+e.target.value)}
              style={{width:160,accentColor:C.accent}} />
          </div>

          <button onClick={()=>{runPrioritize();setShowConfig(false)}}
            disabled={loading.report} style={{
            padding:'8px 20px',borderRadius:8,border:'none',cursor:'pointer',
            background:C.accent,color:'#fff',fontSize:13,fontWeight:600,
            opacity:loading.report?0.6:1,alignSelf:'flex-end',
          }}>
            {loading.report ? 'Running…' : '▶ Run pipeline'}
          </button>
        </div>
      )}

      {/* ── About ── */}
      <div style={{background:C.surface, borderBottom:`1px solid ${C.border}`, padding:'18px 24px'}}>
        <p style={{color:C.textSec, lineHeight:1.7, fontSize:13, margin:'0 0 12px', maxWidth:760}}>
          VRTQ-RL is a self-optimizing test prioritization system that extends the VRTQ
          heuristic framework (Value, Risk, Time, Quality) with reinforcement learning.
          A four-agent pipeline analyzes git diffs, scores risk, and selects the
          highest-value tests to run — improving via a PPO agent trained against
          historical fault data. This dashboard visualizes live prioritization runs and
          compares RL against heuristic and random baselines.
        </p>
        <div style={{display:'flex', gap:24, flexWrap:'wrap'}}>
          {Object.entries(TECH_STACK).map(([cat, items]) => (
            <div key={cat}>
              <div style={{fontSize:10,color:C.textMut,textTransform:'uppercase',
                letterSpacing:'0.06em',marginBottom:6}}>{cat}</div>
              <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
                {items.map(i => (
                  <span key={i} style={{background:C.card,color:C.textSec,
                    border:`1px solid ${C.border}`,borderRadius:99,
                    padding:'2px 9px',fontSize:11}}>{i}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{padding:'24px',opacity:animIn?1:0,
        transform:animIn?'none':'translateY(8px)',transition:'all 0.3s'}}>

        {loading.report && tab==='overview' && (
          <div style={{textAlign:'center',padding:'40px 0',color:C.textMut,fontSize:13}}>
            Running VRTQ-RL pipeline…
          </div>
        )}

        {!loading.report && (
          <>
            {tab==='overview'  && <OverviewTab  report={report} />}
            {tab==='tests'     && <TestQueueTab report={report} />}
            {tab==='compare'   && (
              <BaselinesTab comparison={comparison}
                onRefresh={runCompare} loading={loading.compare} />
            )}
            {tab==='training'  && (
              <TrainingTab curveData={curve.data} source={curve.source}
                onRefresh={runCurve} loading={loading.curve} />
            )}
            {tab==='architecture' && <ArchitectureTab />}
            {tab==='agentic' && (
              <AgentChatTab agenticStatus={agenticStatus} onRun={runAgentic}
                result={agenticResult} loading={loading.agentic} />
            )}
          </>
        )}

        <div style={{marginTop:32, paddingTop:16, borderTop:`1px solid ${C.border}`,
          textAlign:'center', fontSize:11, color:C.textMut}}>
          Built by Anthony Seumal — AIM PhD in Data Science Scholar
        </div>
      </div>
    </div>
  )
}
