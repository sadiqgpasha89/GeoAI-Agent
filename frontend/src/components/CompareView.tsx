import { useState } from 'react'
import { formatDate } from '../lib'
import type { Scene } from '../types'

interface Props { scene: Scene; before: string | null; after: string | null }

export default function CompareView({ scene, before, after }: Props) {
  const [pos, setPos] = useState(50)
  if (!scene.assets.before) return <div className="empty-state">Add an earlier image to compare acquisitions.</div>
  return (
    <div className="compare" style={{ aspectRatio: `${scene.width} / ${scene.height}` }}>
      {after && <img src={after} alt="Latest imagery" draggable={false} />}
      {before && (
        <div className="compare-before" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={before} alt="Earlier imagery" draggable={false} />
        </div>
      )}
      <div className="compare-handle" style={{ left: `${pos}%` }} aria-hidden />
      <span className="compare-tag left">Earlier · {formatDate(scene.before_at)}</span>
      <span className="compare-tag right">Latest · {formatDate(scene.after_at)}</span>
      <input className="compare-range" type="range" min={0} max={100} value={pos} onChange={(e) => setPos(Number(e.target.value))} aria-label="Comparison slider" />
    </div>
  )
}
