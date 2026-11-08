import { useMemo } from 'react'
import { colorFor, polygonPoints, prettyLabel } from '../lib'
import type { Observation, Scene } from '../types'

interface Props {
  scene: Scene
  src: string | null
  observations: Observation[]
  hidden: Set<string>
  selectedId: string | null
  onSelect: (id: string | null) => void
  showOverlay: boolean
  caption?: string
}

function niceLength(maxM: number): number {
  const pow = 10 ** Math.floor(Math.log10(maxM))
  for (const m of [5, 2, 1]) if (m * pow <= maxM) return m * pow
  return pow
}

export default function ImageStage({ scene, src, observations, hidden, selectedId, onSelect, showOverlay, caption }: Props) {
  const visible = useMemo(() => observations.filter((o) => !hidden.has(o.label) && polygonPoints(o)), [observations, hidden])
  const scale = useMemo(() => {
    if (!scene.gsd_m) return null
    const widthM = scene.width * scene.gsd_m
    const len = niceLength(widthM * 0.22)
    return { len, pct: (len / widthM) * 100 }
  }, [scene])
  const sel = observations.find((o) => o.id === selectedId)

  return (
    <div className="stage" style={{ aspectRatio: `${scene.width} / ${scene.height}` }}>
      {src ? <img src={src} alt={`${scene.title} imagery`} draggable={false} /> : <div className="stage-loading">Loading imagery…</div>}
      {showOverlay && (
        <svg className="overlay" viewBox={`0 0 ${scene.width} ${scene.height}`} preserveAspectRatio="none" role="group" aria-label="Evidence overlay" onClick={() => onSelect(null)}>
          {visible.map((o) => {
            const c = colorFor(o)
            const active = o.id === selectedId
            const dim = selectedId && !active
            return (
              <polygon key={o.id} points={polygonPoints(o)!} fill={c} fillOpacity={active ? 0.38 : dim ? 0.07 : 0.2} stroke={c} strokeWidth={active ? 3.2 : 1.8}
                strokeOpacity={dim ? 0.35 : 1} vectorEffect="non-scaling-stroke" className="evidence" tabIndex={0} role="button"
                aria-label={`${prettyLabel(o.label)}, ${Math.round(o.confidence * 100)} percent confidence`}
                onClick={(e) => { e.stopPropagation(); onSelect(active ? null : o.id) }}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(active ? null : o.id) } }}>
                <title>{`${prettyLabel(o.label)} · ${Math.round(o.confidence * 100)}% · ${o.kind}`}</title>
              </polygon>
            )
          })}
        </svg>
      )}
      {showOverlay && sel?.bbox_px && (
        <div className="tag" style={{ left: `${(sel.bbox_px[0] / scene.width) * 100}%`, top: `${(sel.bbox_px[1] / scene.height) * 100}%`, background: colorFor(sel) }}>
          {prettyLabel(sel.label)} · {Math.round(sel.confidence * 100)}%
        </div>
      )}
      {scene.crs && <div className="north" aria-hidden>N<b>↑</b></div>}
      {scale && (
        <div className="scalebar" aria-label={`Scale bar ${scale.len} metres`}>
          <span style={{ width: `${scale.pct}%` }} />
          <small>{scale.len >= 1000 ? `${scale.len / 1000} km` : `${scale.len} m`}</small>
        </div>
      )}
      {caption && <div className="stage-caption">{caption}</div>}
    </div>
  )
}
