import { useState } from 'react'
import { colorFor, download, formatDate, formatMeasure, pctText, prettyLabel, toCSV, toGeoJSON } from '../lib'
import type { AnalysisResult } from '../types'

type Tab = 'findings' | 'metrics' | 'notes' | 'provenance'
interface Props {
  result: AnalysisResult | null
  selectedId: string | null
  onSelect: (id: string | null) => void
  onReview: (decision: 'approved' | 'rejected' | 'corrected') => Promise<void>
  reviewNote: string
}

export default function ResultTabs({ result, selectedId, onSelect, onReview, reviewNote }: Props) {
  const [tab, setTab] = useState<Tab>('findings')
  const [busy, setBusy] = useState(false)
  if (!result) return <div className="panel-card empty-state">Run an analysis to see measurements, evidence and provenance here.</div>
  const base = `geoai-${result.id}`
  const decide = async (d: 'approved' | 'rejected' | 'corrected') => { setBusy(true); try { await onReview(d) } finally { setBusy(false) } }
  const tabs: [Tab, string][] = [['findings', `Findings (${result.observations.length})`], ['metrics', `Measurements (${Object.keys(result.metrics).length})`], ['notes', 'Assumptions & limits'], ['provenance', 'Provenance']]

  return (
    <section className="panel-card result-tabs" aria-label="Analysis details">
      <div className="tabs-head">
        <div role="tablist" aria-label="Result sections">
          {tabs.map(([k, l]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{l}</button>)}
        </div>
        <div className="exports">
          <button className="text-button" onClick={() => download(`${base}.geojson`, JSON.stringify(toGeoJSON(result), null, 1), 'application/geo+json')}>GeoJSON</button>
          <button className="text-button" onClick={() => download(`${base}.csv`, toCSV(result), 'text/csv')}>CSV</button>
          <button className="text-button" onClick={() => download(`${base}.json`, JSON.stringify(result, null, 1), 'application/json')}>JSON</button>
        </div>
      </div>

      {result.needs_review && (
        <div className={`review-banner ${result.review_status}`} role="status">
          <div>
            <strong>{result.review_status === 'pending' ? 'Analyst review required' : `Review decision: ${result.review_status}`}</strong>
            <ul>{result.review_reasons.map((r) => <li key={r}>{r}</li>)}</ul>
            <small>{reviewNote}</small>
          </div>
          {result.review_status === 'pending' && (
            <div className="review-actions">
              <button disabled={busy} onClick={() => decide('approved')}>Approve</button>
              <button disabled={busy} className="ghost" onClick={() => decide('corrected')}>Mark corrected</button>
              <button disabled={busy} className="ghost danger" onClick={() => decide('rejected')}>Reject</button>
            </div>
          )}
        </div>
      )}

      {tab === 'findings' && (
        result.observations.length === 0 ? <p className="muted">No evidence regions were produced for this question.</p> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Evidence</th><th>Kind</th><th>Severity</th><th>Confidence</th><th>Area</th><th>Position (px)</th></tr></thead>
              <tbody>
                {result.observations.map((o) => (
                  <tr key={o.id} className={o.id === selectedId ? 'selected' : ''} onClick={() => onSelect(o.id === selectedId ? null : o.id)} tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') onSelect(o.id === selectedId ? null : o.id) }}>
                    <td><i className="swatch" style={{ background: colorFor(o) }} />{prettyLabel(o.label)}{o.attributes.name || o.attributes.feature_id || o.attributes.road ? <small> · {String(o.attributes.name ?? o.attributes.road ?? o.attributes.feature_id)}</small> : null}</td>
                    <td><span className={`kind ${o.kind}`}>{o.kind.replace('_', ' ')}</span></td>
                    <td>{o.severity ? <span className={`sev ${o.severity}`}>{o.severity}</span> : '—'}</td>
                    <td>{pctText(o.confidence)}</td>
                    <td>{o.area_m2 != null ? `${o.area_m2.toLocaleString(undefined, { maximumFractionDigits: 0 })} m²` : o.area_px != null ? `${Math.round(o.area_px)} px²` : '—'}</td>
                    <td className="mono">{o.bbox_px ? o.bbox_px.map((v) => Math.round(v)).join(', ') : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {tab === 'metrics' && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Measurement</th><th>Value</th><th>Method</th></tr></thead>
            <tbody>{Object.entries(result.metrics).map(([k, m]) => <tr key={k}><td>{prettyLabel(k)}</td><td className="mono">{formatMeasure(m)}</td><td className="muted">{m.method ?? ''}</td></tr>)}</tbody>
          </table>
          {Object.keys(result.metrics).length === 0 && <p className="muted">No measurements for this question.</p>}
        </div>
      )}

      {tab === 'notes' && (
        <div className="notes">
          <div><h3>Assumptions</h3><ul>{result.assumptions.length ? result.assumptions.map((a) => <li key={a}>{a}</li>) : <li>None recorded.</li>}</ul></div>
          <div><h3>Limitations</h3><ul>{result.limitations.length ? result.limitations.map((a) => <li key={a}>{a}</li>) : <li>None recorded.</li>}</ul></div>
          <div><h3>Suggested next actions</h3><ul>{result.next_actions.length ? result.next_actions.map((a) => <li key={a}>{a}</li>) : <li>None.</li>}</ul></div>
        </div>
      )}

      {tab === 'provenance' && (
        <div className="notes">
          <div>
            <h3>Sources</h3>
            <ul>{result.sources.map((s) => <li key={s.role}><b>{s.role}</b> · {formatDate(s.captured_at)} · {s.sensor ?? 'sensor n/a'} · {s.georeferenced ? `${s.crs}, ${s.gsd_m?.toFixed(2)} m/px` : 'not georeferenced'}<br /><span className="mono">sha256 {s.sha256?.slice(0, 20)}…</span></li>)}</ul>
          </div>
          <div>
            <h3>Pipeline</h3>
            {result.provenance ? (
              <ul>
                <li>{result.provenance.pipeline_version} · mode <b>{result.provenance.mode}</b></li>
                <li>Models: {Object.entries(result.provenance.models).map(([k, v]) => `${k}=${v}`).join(', ')}</li>
                <li>Tools: {result.provenance.tool_calls.map((t) => `${t.tool} (${t.duration_ms} ms, ${t.status})`).join(', ') || '—'}</li>
                <li className="mono">record sha256 {result.provenance.record_sha256?.slice(0, 24)}…</li>
              </ul>
            ) : <p className="muted">No provenance recorded.</p>}
          </div>
        </div>
      )}
    </section>
  )
}
