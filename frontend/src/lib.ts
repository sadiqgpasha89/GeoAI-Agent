import type { AnalysisResult, Observation, Severity } from './types'

const LABEL_COLORS: Record<string, string> = {
  flood_extent: '#4db8d6', burn_scar: '#d0605a', smoke_plume: '#b9c3cc', active_fire_front: '#ff8a2a', park: '#7fd18b', construction_site: '#e0b050',
  new_development: '#b69af0', green_space_loss: '#e08fb5', zoning_conflict: '#ff6f91', evacuation_zone: '#f0c040', debris_area: '#d6a878',
  field_healthy: '#6fcf8b', field_watch: '#e6c86a', field_stressed: '#f4a16d', field_severe: '#f47772', field_fallow: '#a08f78', stress_anomaly: '#ff8f6b',
}
const SEV_COLORS: Record<Severity, string> = { critical: '#f47772', high: '#f4a16d', moderate: '#e8c26a', low: '#80d59d', info: '#6db9ed' }

export function colorFor(o: Observation): string {
  return LABEL_COLORS[o.label] ?? SEV_COLORS[(o.severity ?? 'info') as Severity] ?? '#6db9ed'
}

export const prettyLabel = (l: string) => l.replace(/_/g, ' ')
export const pctText = (c: number | null | undefined) => (c == null ? '—' : `${Math.round(c * 100)}%`)

export function formatMeasure(m: { value: number | string | null; unit?: string | null }): string {
  if (m.value == null) return '—'
  const v = typeof m.value === 'number' ? m.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : m.value
  return m.unit ? `${v} ${m.unit}` : String(v)
}

export function sceneAreaKm2(width: number, height: number, gsd?: number | null): number | null {
  return gsd ? (width * gsd * height * gsd) / 1e6 : null
}

export function formatDate(iso?: string | null): string {
  if (!iso) return 'no timestamp'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC'
}

export function polygonPoints(o: Observation): string | null {
  if (o.pixel_polygon?.length) return o.pixel_polygon.map(([x, y]) => `${x},${y}`).join(' ')
  if (o.bbox_px) { const [x0, y0, x1, y1] = o.bbox_px; return `${x0},${y0} ${x1},${y0} ${x1},${y1} ${x0},${y1}` }
  return null
}

// ---- client-side exports (work in static and live mode) -------------------------------------
export function toGeoJSON(res: AnalysisResult) {
  return {
    type: 'FeatureCollection',
    name: `geoai-${res.id}`,
    properties: { analysis_id: res.id, question: res.question, crs: 'EPSG:4326 (null geometry = not georeferenced; see bbox_px)' },
    features: res.observations.map((o) => ({
      type: 'Feature', id: o.id, geometry: o.geometry ?? null,
      properties: { id: o.id, label: o.label, kind: o.kind, severity: o.severity, confidence: o.confidence, area_m2: o.area_m2, bbox_px: o.bbox_px, ...o.attributes },
    })),
  }
}

const cell = (v: unknown) => {
  let s = v == null ? '' : String(v)
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}` // neutralise spreadsheet formula injection
  return `"${s.replace(/"/g, '""')}"`
}
export function toCSV(res: AnalysisResult): string {
  const head = ['id', 'label', 'kind', 'severity', 'confidence', 'area_m2', 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'attributes']
  const rows = res.observations.map((o) => [o.id, o.label, o.kind, o.severity ?? '', o.confidence, o.area_m2 ?? '', ...(o.bbox_px ?? ['', '', '', '']), JSON.stringify(o.attributes)].map(cell).join(','))
  return [head.join(','), ...rows].join('\n')
}

export function download(name: string, text: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  document.body.append(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
