import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertCircle, ArrowRight, Check, ChevronDown, CircleHelp, FlaskConical, Grape, LoaderCircle, RotateCcw, Sparkles, Wine } from 'lucide-react'

type Field = { key: string; label: string; unit: string; min: number; max: number; step: number; red: number; white: number; hint: string }
type Prediction = { predicted_tier: string; confidence: number; top_two_margin: number; entropy: number; probabilities: Record<string, number>; notice: string; model_name: string }

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const fields: Field[] = [
  { key: 'fixed acidity', label: 'Fixed acidity', unit: 'g/L', min: 0, max: 30, step: .1, red: 7.4, white: 6.8, hint: 'Non-volatile acids in the wine' },
  { key: 'volatile acidity', label: 'Volatile acidity', unit: 'g/L', min: 0, max: 5, step: .01, red: .70, white: .27, hint: 'Primarily acetic acid content' },
  { key: 'citric acid', label: 'Citric acid', unit: 'g/L', min: 0, max: 5, step: .01, red: 0, white: .36, hint: 'Can add freshness and flavor' },
  { key: 'residual sugar', label: 'Residual sugar', unit: 'g/L', min: 0, max: 100, step: .1, red: 1.9, white: 5.0, hint: 'Sugar remaining after fermentation' },
  { key: 'chlorides', label: 'Chlorides', unit: 'g/L', min: 0, max: 2, step: .001, red: .076, white: .045, hint: 'Salt content' },
  { key: 'free sulfur dioxide', label: 'Free sulfur dioxide', unit: 'mg/L', min: 0, max: 500, step: 1, red: 11, white: 30, hint: 'Free form of sulfur dioxide' },
  { key: 'total sulfur dioxide', label: 'Total sulfur dioxide', unit: 'mg/L', min: 0, max: 1000, step: 1, red: 34, white: 120, hint: 'Free and bound sulfur dioxide' },
  { key: 'density', label: 'Density', unit: 'g/cm³', min: .8, max: 1.2, step: .0001, red: .9978, white: .994, hint: 'Wine density' },
  { key: 'pH', label: 'pH', unit: 'scale', min: 0, max: 14, step: .01, red: 3.51, white: 3.20, hint: 'Acidity / alkalinity measure' },
  { key: 'sulphates', label: 'Sulphates', unit: 'g/L', min: 0, max: 5, step: .01, red: .56, white: .45, hint: 'Sulphate additives' },
  { key: 'alcohol', label: 'Alcohol', unit: '% vol', min: 0, max: 30, step: .1, red: 9.4, white: 10.0, hint: 'Alcohol by volume' },
]
const initial = (kind: 'red' | 'white') => Object.fromEntries(fields.map(f => [f.key, kind === 'red' ? f.red : f.white])) as Record<string, number>

function App() {
  const [wineType, setWineType] = useState<'red' | 'white'>('red')
  const [values, setValues] = useState<Record<string, number>>(initial('red'))
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [modelInfo, setModelInfo] = useState<any>(null)

  useEffect(() => {
    fetch(`${API}/api/model-info`).then(r => r.ok ? r.json() : Promise.reject()).then(d => { setModelInfo(d); setApiOnline(true) }).catch(() => setApiOnline(false))
  }, [])

  const completion = useMemo(() => fields.filter(f => Number.isFinite(values[f.key])).length, [values])
  const changeType = (type: 'red' | 'white') => { setWineType(type); setValues(initial(type)); setResult(null); setError('') }
  const update = (key: string, value: number) => { setValues(prev => ({ ...prev, [key]: value })); setResult(null) }
  const reset = () => { setValues(initial(wineType)); setResult(null); setError('') }

  async function predict() {
    setLoading(true); setError(''); setResult(null)
    try {
      if (values['total sulfur dioxide'] < values['free sulfur dioxide']) throw new Error('Total sulfur dioxide must be greater than or equal to free sulfur dioxide.')
      const payload = { wine_type: wineType, ...values }
      const response = await fetch(`${API}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed. Check the values and try again.')
      setResult(data)
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to reach the prediction service.') }
    finally { setLoading(false) }
  }

  const tierClass = result?.predicted_tier.startsWith('High') ? 'high' : result?.predicted_tier.startsWith('Low') ? 'low' : 'average'

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#top"><span className="brand-mark"><Grape size={20}/></span><span>WineSense<span className="brand-ai"> AI</span></span></a><div className="topbar-right"><span className={`api-status ${apiOnline ? 'online' : 'offline'}`}><i/>API {apiOnline ? 'connected' : 'offline'}</span><a className="docs-link" href={`${API}/docs`} target="_blank" rel="noreferrer">API docs <ArrowRight size={14}/></a></div></header>

    <main id="top">
      <section className="hero"><div className="hero-copy"><div className="eyebrow"><Sparkles size={13}/> MACHINE LEARNING · QUALITY INTELLIGENCE</div><h1>Discover the quality<br/><em>within every drop.</em></h1><p>Explore how physicochemical properties shape wine quality. Enter a wine profile and let our trained machine learning model estimate its quality category.</p><div className="hero-meta"><span><span className="meta-dot"/> Extra Trees Classifier</span><span>·</span><span>17 engineered features</span><span>·</span><span>3 quality tiers</span></div></div><div className="hero-art"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-glass"><Wine size={112} strokeWidth={.85}/></div><span className="art-caption">THE ART OF<br/>PREDICTION</span><span className="art-spark spark-a">✳</span><span className="art-spark spark-b">✦</span></div></section>

      <section className="workspace"><div className="section-heading"><div><div className="eyebrow">01 / INPUT PROFILE</div><h2>Build your wine profile</h2><p>Choose a wine style, then adjust its chemical measurements.</p></div><button className="reset-button" onClick={reset}><RotateCcw size={15}/> Reset values</button></div>

        <div className="wine-switch"><button className={wineType === 'red' ? 'selected red-selected' : ''} onClick={() => changeType('red')}><span className="wine-dot red-dot"/> Red wine</button><button className={wineType === 'white' ? 'selected white-selected' : ''} onClick={() => changeType('white')}><span className="wine-dot white-dot"/> White wine</button><span className="switch-note">Preset values are editable</span></div>

        <div className="input-panel"><div className="input-panel-head"><div><FlaskConical size={17}/><strong>Physicochemical properties</strong></div><span>{completion} / 11 parameters</span></div><div className="field-grid">{fields.map((f, i) => <label className="field" key={f.key}><span className="field-top"><span>{f.label}</span><span className="field-unit">{f.unit}</span></span><span className="field-input-wrap"><input type="number" min={f.min} max={f.max} step={f.step} value={values[f.key]} onChange={e => update(f.key, Number(e.target.value))}/><span className="field-index">{String(i + 1).padStart(2, '0')}</span></span><span className="field-hint">{f.hint}</span></label>)}</div><div className="input-footer"><span><CircleHelp size={14}/> Values should use the units shown above.</span><button className="predict-button" onClick={predict} disabled={loading || !apiOnline}>{loading ? <><LoaderCircle className="spin" size={17}/> Analyzing profile</> : <>Predict quality <ArrowRight size={17}/></>}</button></div>{!apiOnline && <div className="inline-warning"><AlertCircle size={15}/> Start the FastAPI backend to enable predictions. See the setup instructions in the README.</div>}{error && <div className="error-message"><AlertCircle size={15}/>{error}</div>}</div>
      </section>

      <section className="result-section"><div className="section-heading"><div><div className="eyebrow">02 / MODEL OUTPUT</div><h2>Your prediction</h2><p>Model-estimated category and probability distribution.</p></div><span className="result-tag"><Activity size={14}/> LIVE INFERENCE</span></div>
        {!result ? <div className="empty-result"><div className="empty-icon"><Sparkles size={25}/></div><div><strong>Your result will appear here</strong><p>Complete the wine profile and select <b>Predict quality</b> to run the model.</p></div><div className="empty-arrow"><ArrowRight size={20}/></div></div> : <div className="result-grid"><div className={`prediction-card ${tierClass}`}><div className="prediction-card-top"><span>ESTIMATED QUALITY TIER</span><span className="prediction-check"><Check size={15}/></span></div><div className="prediction-main"><div className="prediction-glyph"><Wine size={30}/></div><div><div className="prediction-tier">{result.predicted_tier}</div><div className="prediction-sub">Predicted for {wineType} wine</div></div></div><div className="confidence-row"><span>Top class probability</span><strong>{(result.confidence * 100).toFixed(1)}%</strong></div><div className="confidence-track"><span style={{ width: `${result.confidence * 100}%` }}/></div><div className="prediction-foot">Model: {result.model_name}</div></div><div className="probability-card"><div className="probability-title"><div><strong>Class probabilities</strong><p>Relative model output across all tiers</p></div><span className="probability-symbol">%</span></div>{Object.entries(result.probabilities).map(([label, prob]) => <div className="probability-item" key={label}><div className="probability-label"><span>{label}</span><b>{(prob * 100).toFixed(1)}%</b></div><div className="probability-track"><span className={label.startsWith('Low') ? 'bar-low' : label.startsWith('High') ? 'bar-high' : 'bar-average'} style={{ width: `${prob * 100}%` }}/></div></div>)}<div className="result-insight"><Sparkles size={15}/><span>Top-two probability margin: <b>{(result.top_two_margin * 100).toFixed(1)} percentage points</b></span></div></div><div className="disclaimer"><AlertCircle size={15}/><span>{result.notice}</span></div></div>}
      </section>

      <section className="model-strip"><div className="model-strip-title"><span className="eyebrow">THE ENGINE</span><h3>Built on your trained ML research</h3><p>The prediction service reuses the Extra Trees model and engineered feature definitions from your Jupyter notebook.</p></div><div className="model-stats"><div><b>{modelInfo?.training_rows?.toLocaleString() ?? '5,320'}</b><span>Deduplicated samples</span></div><div><b>{modelInfo?.features ?? 17}</b><span>Input features</span></div><div><b>{modelInfo?.metrics?.test_accuracy ? `${(modelInfo.metrics.test_accuracy * 100).toFixed(2)}%` : '74.91%'}</b><span>Test accuracy</span></div></div></section>
    </main>
    <footer><a className="brand footer-brand" href="#top"><span className="brand-mark"><Grape size={17}/></span><span>WineSense<span className="brand-ai"> AI</span></span></a><span>Intelligent Wine Quality Prediction & Multidimensional Analysis</span><span>Built with React · FastAPI · Scikit-learn</span></footer>
  </div>
}
export default App
