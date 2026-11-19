import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createLiveBackend, createStaticBackend, type LiveConfig } from './api'
import AgentPanel, { type Msg } from './components/AgentPanel'
import CompareView from './components/CompareView'
import ImageStage from './components/ImageStage'
import MapView from './components/MapView'
import ResultTabs from './components/ResultTabs'
import SettingsDialog from './components/SettingsDialog'
import { colorFor, formatDate, pctText, prettyLabel, sceneAreaKm2 } from './lib'
import type { AnalysisResult, Domain, Scene, TraceStep } from './types'

const DOMAINS: { id: Domain; label: string; dot: string; placeholder: string }[] = [
  { id: 'flood', label: 'Flood monitoring', dot: 'flood-dot', placeholder: 'Ask about water extent, flooded buildings or roads…' },
  { id: 'wildfire', label: 'Wildfire monitoring', dot: 'fire-dot', placeholder: 'Ask about burn scars, smoke or exposure…' },
  { id: 'earthquake', label: 'Earthquake impact', dot: 'quake-dot', placeholder: 'Ask about damage, debris or blocked roads…' },
  { id: 'urban', label: 'Urban planning', dot: 'city-dot', placeholder: 'Ask about land use, zoning or park access…' },
  { id: 'agriculture', label: 'Crop health', dot: 'crop-dot', placeholder: 'Ask about fields, stress or trends…' },
]
const UPLOAD_QUESTIONS: Record<Domain, string[]> = {
  flood: ['Describe this scene in detail', 'Is there flooding in this image?', 'Locate the water'],
  wildfire: ['Describe this scene in detail', 'Where are the burn scars?', 'Locate the smoke plume'],
  earthquake: ['Describe this scene in detail', 'Which buildings look damaged?', 'Are any roads blocked by debris?'],
  urban: ['Describe this scene in detail', 'How is land used in this image?', 'Locate the parks'],
  agriculture: ['Describe this scene in detail', 'Which areas show crop stress?', 'Estimate area of stressed crops'],
}
const STORE_KEY = 'terralens.live'
const loadCfg = (): LiveConfig | null => {
  const envUrl = import.meta.env.VITE_API_URL as string | undefined
  try { const raw = localStorage.getItem(STORE_KEY); if (raw) return JSON.parse(raw) as LiveConfig } catch { /* storage unavailable */ }
  return envUrl ? { baseUrl: envUrl, apiKey: '' } : null
}
let nid = 0
const uid = () => `m${++nid}-${Date.now()}`

export default function App() {
  const [cfg, setCfg] = useState<LiveConfig | null>(loadCfg)
  const backend = useMemo(() => (cfg ? createLiveBackend(cfg) : createStaticBackend()), [cfg])
  const [scenes, setScenes] = useState<Scene[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [view, setView] = useState<'image' | 'map' | 'compare'>('image')
  const [showBefore, setShowBefore] = useState(false)
  const [imgs, setImgs] = useState<{ after: string | null; before: string | null }>({ after: null, before: null })
  const [msgs, setMsgs] = useState<Record<string, Msg[]>>({})
  const [results, setResults] = useState<Record<string, AnalysisResult | null>>({})
  const [history, setHistory] = useState<{ sceneId: string; result: AnalysisResult }[]>([])
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [overlay, setOverlay] = useState(true)
  const [busy, setBusy] = useState(false)
  const [settings, setSettings] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const bootRan = useRef(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const beforeRef = useRef<HTMLInputElement>(null)

  const scene = scenes.find((s) => s.id === activeId) ?? null
  const result = scene ? results[scene.id] ?? null : null
  const say = useCallback((m: string) => { setToast(m); window.setTimeout(() => setToast(null), 3500) }, [])

  // ---- load scenes -----------------------------------------------------
  useEffect(() => {
    let off = false
    setScenes([]); setActiveId(null); setLoadError(null); setMsgs({}); setResults({}); setHistory([]); bootRan.current = false
    backend.listScenes().then((raw) => {
      if (off) return
      const order = DOMAINS.map((d) => d.id)
      const s = [...raw].sort((a, b) => order.indexOf(a.domain) - order.indexOf(b.domain))
      setScenes(s); setActiveId(s[0]?.id ?? null)
    })
      .catch((e: Error) => { if (!off) setLoadError(e.message) })
    return () => { off = true }
  }, [backend])

  // ---- load imagery for the active scene -------------------------------
  useEffect(() => {
    if (!scene) return
    let off = false
    setImgs({ after: null, before: null }); setSelectedId(null); setHidden(new Set()); setShowBefore(false)
    backend.imageUrl(scene, 'after').then((u) => !off && setImgs((p) => ({ ...p, after: u }))).catch((e: Error) => say(e.message))
    if (scene.assets.before) backend.imageUrl(scene, 'before').then((u) => !off && setImgs((p) => ({ ...p, before: u }))).catch(() => undefined)
    return () => { off = true }
  }, [backend, scene?.id, scene?.assets.before]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- ask the agent ---------------------------------------------------
  const ask = useCallback(async (question: string, example = false) => {
    if (!scene || busy) return
    const sid = scene.id
    const pendingId = uid()
    const add = (m: Msg) => setMsgs((p) => ({ ...p, [sid]: [...(p[sid] ?? []), m] }))
    const patch = (id: string, f: (m: Msg) => Msg) => setMsgs((p) => ({ ...p, [sid]: (p[sid] ?? []).map((m) => (m.id === id ? f(m) : m)) }))
    add({ id: uid(), role: 'user', text: question, example })
    add({ id: pendingId, role: 'agent', text: '', pending: true, steps: [] })
    setBusy(true)
    try {
      const res = await backend.analyze(scene, question, {
        onStep: (s: TraceStep) => patch(pendingId, (m) => ({ ...m, steps: [...(m.steps ?? []), s], tool: undefined })),
        onToolStart: (tool) => patch(pendingId, (m) => ({ ...m, tool })),
      })
      patch(pendingId, (m) => ({ ...m, pending: false, text: res.answer || res.clarification || 'No answer was produced.', result: res }))
      setResults((p) => ({ ...p, [sid]: res }))
      if (res.observations.length) setHistory((h) => [{ sceneId: sid, result: res }, ...h].slice(0, 20))
      setSelectedId(null); setHidden(new Set()); setView((v) => (v === 'compare' ? 'image' : v))
    } catch (e) {
      patch(pendingId, (m) => ({ ...m, pending: false, error: true, text: (e as Error).message || 'The analysis failed.' }))
    } finally { setBusy(false) }
  }, [backend, scene, busy])

  // static demo: run one example so the first screen is not empty
  useEffect(() => {
    if (backend.kind !== 'static' || !scene || bootRan.current) return
    bootRan.current = true
    void ask(scene.suggested[3] ?? scene.suggested[0], true)
  }, [backend.kind, scene, ask])

  // ---- uploads (live backend) -----------------------------------------
  const addScene = (s: Scene) => { setScenes((p) => [...p, s]); setActiveId(s.id); setView('image') }
  const onUpload = async (file: File | undefined) => {
    if (!file) return
    if (!backend.canUpload) { setSettings(true); return }
    try {
      say(`Uploading ${file.name}…`)
      const a = await backend.upload(file)
      const domain = scene?.domain ?? 'flood'
      addScene({ id: `upload-${a.id}`, domain, title: file.name, location: 'Uploaded image', description: 'User-supplied imagery', width: a.width, height: a.height, crs: a.crs ?? null,
        gsd_m: a.gsd_m ?? null, bounds_wgs84: a.bounds_wgs84 ?? null, after_at: a.captured_at ?? undefined, suggested: UPLOAD_QUESTIONS[domain], assets: { after: a.id }, upload: true })
      say(a.georeferenced ? 'Georeferenced image loaded' : 'Image loaded (not georeferenced: measurements are in pixels)')
    } catch (e) { say((e as Error).message) }
  }
  const onBefore = async (file: File | undefined) => {
    if (!file || !scene) return
    try {
      const a = await backend.upload(file)
      setScenes((p) => p.map((s) => (s.id === scene.id ? { ...s, assets: { ...s.assets, before: a.id }, before_at: a.captured_at ?? undefined } : s)))
      say('Earlier image attached — change analysis enabled')
    } catch (e) { say((e as Error).message) }
  }

  const pickDomain = (d: Domain) => {
    if (scene?.upload) { setScenes((p) => p.map((s) => (s.id === scene.id ? { ...s, domain: d, suggested: UPLOAD_QUESTIONS[d] } : s))); return }
    const target = scenes.find((s) => s.domain === d && !s.upload)
    if (target) { setActiveId(target.id); setView('image') }
  }

  const review = async (decision: 'approved' | 'rejected' | 'corrected') => {
    if (!result) return
    try {
      const status = await backend.review(result, decision)
      const upd = (r: AnalysisResult) => (r.id === result.id ? { ...r, review_status: status } : r)
      setResults((p) => ({ ...p, [scene!.id]: upd(result) }))
      setHistory((h) => h.map((x) => ({ ...x, result: upd(x.result) })))
      setMsgs((p) => ({ ...p, [scene!.id]: (p[scene!.id] ?? []).map((m) => (m.result ? { ...m, result: upd(m.result) } : m)) }))
      say(`Review recorded: ${status}`)
    } catch (e) { say((e as Error).message) }
  }

  const saveCfg = (c: LiveConfig | null) => {
    try { c ? localStorage.setItem(STORE_KEY, JSON.stringify(c)) : localStorage.removeItem(STORE_KEY) } catch { /* ignore */ }
    setCfg(c); setSettings(false)
  }

  // ---- derived UI data -------------------------------------------------
  const labelGroups = useMemo(() => {
    const m = new Map<string, { n: number; color: string }>()
    result?.observations.forEach((o) => m.set(o.label, { n: (m.get(o.label)?.n ?? 0) + 1, color: m.get(o.label)?.color ?? colorFor(o) }))
    return [...m.entries()]
  }, [result])
  const priority = useMemo(() => (result?.observations ?? []).filter((o) => o.severity === 'critical' || o.severity === 'high').slice(0, 6), [result])
  const crit = (result?.observations ?? []).filter((o) => o.severity === 'critical').length
  const pendingReviews = history.filter((h) => h.result.review_status === 'pending').length
  const area = scene ? sceneAreaKm2(scene.width, scene.height, scene.gsd_m) : null
  const domainCfg = DOMAINS.find((d) => d.id === scene?.domain) ?? DOMAINS[0]
  const messages = scene ? msgs[scene.id] ?? [] : []
  const welcome: Msg[] = scene ? [{ id: 'welcome', role: 'agent', text: `I'm ready to investigate ${scene.title}. Ask a question, or pick a suggestion — every answer is backed by measurements, evidence regions and provenance.` }] : []
  const toggleLabel = (l: string) => setHidden((p) => { const n = new Set(p); n.has(l) ? n.delete(l) : n.add(l); return n })
  const shownSrc = showBefore && imgs.before ? imgs.before : imgs.after

  if (loadError) return <div className="fatal"><h1>Could not load TerraLens</h1><p>{loadError}</p><button className="outline-button" onClick={() => setSettings(true)}>Connection settings</button><SettingsDialog open={settings} initial={cfg} onClose={() => setSettings(false)} onSave={saveCfg} /></div>

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span /><span /><span /></div>
          <div><strong>TerraLens</strong><small>GeoAI agent</small></div>
        </div>
        <div className="workspace-label domain-label">MONITORING DOMAINS</div>
        <nav className="domain-list" aria-label="Monitoring domains">
          {DOMAINS.map((d) => (
            <button key={d.id} className={`domain ${scene?.domain === d.id ? 'active' : ''}`} onClick={() => pickDomain(d.id)} aria-current={scene?.domain === d.id}>
              <i className={`domain-dot ${d.dot}`} /><span>{d.label}</span>
            </button>
          ))}
        </nav>
        <div className="workspace-label scene-label">SCENES</div>
        <div className="scene-list">
          {scenes.filter((s) => s.upload || s.domain === scene?.domain).map((s) => (
            <button key={s.id} className={`scene-item ${s.id === activeId ? 'active' : ''}`} onClick={() => { setActiveId(s.id); setView('image') }}>{s.upload ? '⬆ ' : ''}{s.title}</button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <button className="system-status" onClick={() => setSettings(true)} aria-label="Connection settings">
            <span className="pulse" /><span>{backend.label}</span>
            <small>{backend.kind === 'static' ? 'Pre-computed by the agent · click to connect an API' : cfg?.baseUrl}</small>
          </button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs"><span>TerraLens</span><b>/</b><strong>{domainCfg.label}</strong></div>
          <div className="top-actions">
            <span className="updated"><span className="status-dot" />{backend.kind === 'static' ? 'Synthetic sample scenes' : 'Connected'}</span>
            <a className="text-button" href="https://sadiqgpasha89.github.io/" target="_blank" rel="noreferrer" title="Return to Sadiq G Pasha Portfolio" style={{fontWeight: 700, color: "var(--accent, #38bdf8)"}}>← Portfolio</a>
            <a className="text-button" href="https://github.com/sadiqgpasha89/GeoAI-Agent" target="_blank" rel="noreferrer">GitHub ↗</a>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <p className="eyebrow">AGENTIC GEOSPATIAL INTELLIGENCE · <span>{(scene?.domain ?? 'flood').toUpperCase()}</span></p>
            <h1>Ask your imagery <span>✦</span></h1>
            <p className="heading-copy">An LLM + vision-language agent plans the analysis; deterministic geospatial tools measure it; every answer links back to evidence.</p>
          </div>
          <div className="heading-actions">
            <button className="outline-button" onClick={() => (backend.canUpload ? fileRef.current?.click() : setSettings(true))} title={backend.canUpload ? 'Upload imagery' : 'Uploads need the live API'}><span>＋</span> New analysis</button>
            <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp,image/tiff" hidden onChange={(e) => { void onUpload(e.target.files?.[0]); e.target.value = '' }} />
            <input ref={beforeRef} type="file" accept="image/png,image/jpeg,image/webp,image/tiff" hidden onChange={(e) => { void onBefore(e.target.files?.[0]); e.target.value = '' }} />
          </div>
        </section>

        <section className="stats-grid" aria-label="Situation summary">
          <article className="stat-card"><div className="stat-icon blue">◒</div><div><span>Area covered</span><strong>{area != null ? area.toFixed(area < 1 ? 2 : 1) : '—'} <small>{area != null ? 'km²' : 'not georeferenced'}</small></strong><em>{scene?.gsd_m ? `${scene.gsd_m.toFixed(2)} m/px` : 'pixel units'}</em></div></article>
          <article className="stat-card"><div className="stat-icon orange">⌁</div><div><span>Evidence regions</span><strong>{result?.observations.length ?? '—'} <small>{crit ? `${crit} critical` : 'regions'}</small></strong><em>{result ? `${result.metrics ? Object.keys(result.metrics).length : 0} measurements` : 'run an analysis'}</em></div></article>
          <article className="stat-card"><div className="stat-icon green">◉</div><div><span>Review queue</span><strong>{pendingReviews} <small>pending</small></strong><em>this session</em></div></article>
          <article className="stat-card"><div className="stat-icon purple">✦</div><div><span>Agent confidence</span><strong>{pctText(result?.confidence)}</strong><em>mean of evidence confidence</em></div></article>
        </section>

        <section className="analysis-layout">
          <div className="visual-column">
            <div className="section-header">
              <div><h2>Analysis workspace</h2><p>{scene ? `${scene.title} · ${scene.location}` : 'Loading…'}</p></div>
              <div className="view-controls" role="tablist" aria-label="View">
                {(['image', 'map', 'compare'] as const).map((v) => <button key={v} role="tab" aria-selected={view === v} className={`view-control ${view === v ? 'active' : ''}`} onClick={() => setView(v)}>{{ image: '▧ Image', map: '⌖ Map', compare: '◴ Compare' }[v]}</button>)}
              </div>
            </div>
            <div className="image-card">
              <div className="image-toolbar">
                <div className="image-title">
                  {busy ? <span className="live-badge"><i /> ANALYSING</span> : <span className="live-badge idle">{result ? (result.mode === 'live' ? 'LIVE MODEL' : 'MEASURED') : 'READY'}</span>}
                  <strong>{scene?.title ?? '—'}</strong>
                  <small>{scene?.sensor ?? 'Uploaded image'} · {formatDate(showBefore ? scene?.before_at : scene?.after_at)}</small>
                </div>
                <div className="toolbar-actions">
                  {scene?.assets.before && view === 'image' && <button className={showBefore ? 'on' : ''} onClick={() => setShowBefore((v) => !v)} aria-pressed={showBefore} title="Toggle earlier image">{showBefore ? 'Earlier' : 'Latest'}</button>}
                  <button className={overlay ? 'on' : ''} onClick={() => setOverlay((v) => !v)} aria-pressed={overlay} title="Toggle evidence overlay">◈ Evidence</button>
                  {scene?.upload && !scene.assets.before && <button onClick={() => beforeRef.current?.click()} title="Attach an earlier image">＋ Earlier</button>}
                </div>
              </div>
              <div className="view-body">
                {scene && view === 'image' && <ImageStage scene={scene} src={shownSrc} observations={result?.observations ?? []} hidden={hidden} selectedId={selectedId} onSelect={setSelectedId} showOverlay={overlay && !showBefore} caption={showBefore ? 'Earlier acquisition (no overlay)' : undefined} />}
                {scene && view === 'map' && <MapView scene={scene} src={imgs.after} observations={result?.observations ?? []} hidden={hidden} layersUrl={backend.layersUrl?.(scene)} selectedId={selectedId} />}
                {scene && view === 'compare' && <CompareView scene={scene} before={imgs.before} after={imgs.after} />}
              </div>
              {labelGroups.length > 0 && (
                <div className="legend-row" aria-label="Evidence layers">
                  {labelGroups.map(([l, g]) => (
                    <button key={l} className={`chip ${hidden.has(l) ? 'off' : ''}`} onClick={() => toggleLabel(l)} aria-pressed={!hidden.has(l)}><i style={{ background: g.color }} />{prettyLabel(l)} <b>{g.n}</b></button>
                  ))}
                </div>
              )}
              <div className="image-footer">
                <span>{scene?.crs ? `${scene.crs} · ` : ''}{scene ? `${scene.width}×${scene.height} px` : ''}</span>
                <span>Coordinates: pixels (origin top-left){scene?.crs ? ' + WGS84 geometry' : ' only'}</span>
              </div>
            </div>
          </div>

          <AgentPanel messages={[...welcome, ...messages]} suggestions={scene?.suggested ?? []} busy={busy} placeholder={domainCfg.placeholder} backendLabel={backend.label}
            onAsk={(q) => void ask(q)} onSelectResult={(r) => { if (scene) setResults((p) => ({ ...p, [scene.id]: r })); setSelectedId(null); setView('image'); setShowBefore(false) }} />
        </section>

        <ResultTabs result={result} selectedId={selectedId} onSelect={setSelectedId} onReview={review}
          reviewNote={backend.kind === 'static' ? 'Static demo: decisions are kept in this browser session only.' : 'Decisions are stored as an append-only audit record.'} />

        <section className="bottom-grid">
          <article className="panel-card">
            <div className="panel-title"><div><h2>Recent analyses</h2><p>This session</p></div></div>
            {history.length === 0 ? <p className="muted">Nothing yet — ask a question.</p> : history.slice(0, 5).map((h) => (
              <button key={h.result.id + h.sceneId} className="activity-item" onClick={() => { setActiveId(h.sceneId); setResults((p) => ({ ...p, [h.sceneId]: h.result })); setView('image') }}>
                <span className="activity-icon">⌁</span><div><strong>{h.result.question}</strong><small>{scenes.find((s) => s.id === h.sceneId)?.title} · {h.result.observations.length} regions</small></div>
                <span className={`confidence ${(h.result.confidence ?? 0) >= 0.75 ? 'high' : 'medium'}`}>{pctText(h.result.confidence)}</span>
              </button>
            ))}
          </article>
          <article className="panel-card">
            <div className="panel-title"><div><h2>Priority findings</h2><p>Critical and high-severity evidence in the current result</p></div></div>
            {priority.length === 0 ? <p className="muted">No critical or high-severity findings in the current result.</p> : priority.map((o) => (
              <button key={o.id} className="alert-item" onClick={() => { setSelectedId(o.id); setView('image'); setShowBefore(false) }}>
                <span className={`alert-mark ${o.severity}`}>!</span><div><strong>{prettyLabel(o.label)}</strong><small>{String(o.attributes.name ?? o.attributes.road ?? o.attributes.feature_id ?? o.id)} · {pctText(o.confidence)} · {o.kind}</small></div>
                <span className={`severity ${o.severity}`}>{o.severity}</span>
              </button>
            ))}
          </article>
        </section>
        <footer className="app-footer">
          <span>TerraLens v1.0 · synthetic sample imagery · image-derived estimates are not official emergency measurements</span>
          <span>{result?.provenance ? `${result.provenance.pipeline_version} · ${Object.values(result.provenance.models)[0]}` : 'LangGraph agent · FastAPI · rasterio/shapely'}</span>
        </footer>
      </main>
      <SettingsDialog open={settings} initial={cfg} onClose={() => setSettings(false)} onSave={saveCfg} />
      <div className={`toast ${toast ? 'show' : ''}`} role="status">{toast}</div>
    </div>
  )
}
