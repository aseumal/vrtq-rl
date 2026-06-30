export const C = {
  bg: '#ffffff', surface: '#f8fafc', card: '#ffffff', border: '#e2e8f0',
  accent: '#2563eb', accentLo: '#eff6ff', teal: '#0f766e', amber: '#b45309',
  red: '#dc2626', green: '#15803d', purple: '#7c3aed',
  textPri: '#0f172a', textSec: '#475569', textMut: '#64748b',
  tealBg: '#ccfbf1', purpleBg: '#ede9fe',
  gridLine: '#e5e7eb', axisTick: '#64748b',
  tooltipBg: '#ffffff', tooltipBorder: '#cbd5e1',
}

export const METHOD_COLOR = {
  'Random': '#64748b',
  'VRTQ Heuristic': '#b45309',
  'DQN': '#7c3aed',
  'PPO (VRTQ-RL)': '#0f766e',
}

export const MODULES = [
  'payment_service','auth_service','user_management','order_processing',
  'inventory_service','notification_service','reporting_engine',
  'api_gateway','data_pipeline','search_service',
]

export const pct = (v) => `${(v * 100).toFixed(1)}%`
