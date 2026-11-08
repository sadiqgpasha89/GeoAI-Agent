import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef, useState } from 'react'
import { colorFor, prettyLabel } from '../lib'
import type { Observation, Scene } from '../types'

interface Props { scene: Scene; src: string | null; observations: Observation[]; hidden: Set<string>; layersUrl?: string; selectedId: string | null }

export default function MapView({ scene, src, observations, hidden, layersUrl, selectedId }: Props) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const overlay = useRef<L.ImageOverlay | null>(null)
  const obsLayer = useRef<L.GeoJSON | null>(null)
  const refLayer = useRef<L.GeoJSON | null>(null)
  const [opacity, setOpacity] = useState(0.9)
  const [showRef, setShowRef] = useState(false)
  const b = scene.bounds_wgs84

  useEffect(() => {
    if (!el.current || !b) return
    const m = L.map(el.current, { zoomControl: true, attributionControl: true, scrollWheelZoom: false })
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' }).addTo(m)
    m.fitBounds([[b[1], b[0]], [b[3], b[2]]])
    map.current = m
    return () => { m.remove(); map.current = null; overlay.current = null; obsLayer.current = null; refLayer.current = null }
  }, [scene.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const m = map.current
    if (!m || !b || !src) return
    overlay.current?.remove()
    overlay.current = L.imageOverlay(src, [[b[1], b[0]], [b[3], b[2]]], { opacity }).addTo(m)
    overlay.current.bringToBack()
  }, [src, scene.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { overlay.current?.setOpacity(opacity) }, [opacity])

  useEffect(() => {
    const m = map.current
    if (!m) return
    obsLayer.current?.remove()
    const feats = observations.filter((o) => o.geometry && !hidden.has(o.label)).map((o) => ({
      type: 'Feature' as const, geometry: o.geometry!, properties: { id: o.id, label: o.label, confidence: o.confidence, color: colorFor(o) },
    }))
    obsLayer.current = L.geoJSON({ type: 'FeatureCollection', features: feats } as GeoJSON.FeatureCollection, {
      style: (f) => ({ color: f?.properties.color, weight: f?.properties.id === selectedId ? 4 : 2, fillOpacity: f?.properties.id === selectedId ? 0.4 : 0.2 }),
      onEachFeature: (f, layer) => layer.bindTooltip(`${prettyLabel(f.properties.label)} · ${Math.round(f.properties.confidence * 100)}%`),
    }).addTo(m)
  }, [observations, hidden, selectedId])

  useEffect(() => {
    const m = map.current
    refLayer.current?.remove(); refLayer.current = null
    if (!m || !showRef || !layersUrl) return
    let cancelled = false
    fetch(layersUrl).then((r) => r.json()).then((fc) => {
      if (cancelled || !map.current) return
      refLayer.current = L.geoJSON(fc, { style: (f) => ({ color: f?.properties.layer === 'roads' ? '#ffd166' : '#ffffff', weight: 1.2, fillOpacity: 0.04, dashArray: '4 3' }) }).addTo(map.current)
    }).catch(() => undefined)
    return () => { cancelled = true }
  }, [showRef, layersUrl, scene.id])

  if (!b) return <div className="empty-state">This image is not georeferenced, so it cannot be placed on a map. Measurements are in pixels.</div>
  return (
    <div className="mapwrap">
      <div ref={el} className="map" role="application" aria-label="Map view" />
      <div className="map-controls">
        <label>Imagery opacity <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} /></label>
        {layersUrl && <label className="check"><input type="checkbox" checked={showRef} onChange={(e) => setShowRef(e.target.checked)} /> Reference layers</label>}
        <small>Synthetic scene placed at arbitrary coordinates (EPSG:4326 display).</small>
      </div>
    </div>
  )
}
