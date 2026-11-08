// Mirrors backend/src/geoai/schemas.py. Pixel coordinates: origin top-left, x right, y down.
export type Domain = 'flood' | 'wildfire' | 'earthquake' | 'urban' | 'agriculture'
export type Severity = 'info' | 'low' | 'moderate' | 'high' | 'critical'
export type Kind = 'observed' | 'inferred' | 'not_determinable'

export interface Measurement { value: number | string | null; unit?: string | null; method?: string | null }
export interface Evidence { asset_id: string; region_px?: number[] | null; source_tool: string; note?: string | null }
export interface Observation {
  id: string
  label: string
  kind: Kind
  severity?: Severity | null
  confidence: number
  bbox_px?: number[] | null
  pixel_polygon?: number[][] | null
  geometry?: GeoJSON.Geometry | null // WGS84 lon/lat; null when the asset is not georeferenced
  area_m2?: number | null
  area_px?: number | null
  attributes: Record<string, unknown>
  evidence: Evidence[]
}
export interface Source {
  asset_id: string; role: 'latest' | 'before'; captured_at?: string | null; sensor?: string | null; crs?: string | null
  gsd_m?: number | null; width?: number | null; height?: number | null; sha256?: string | null; georeferenced: boolean
}
export interface TraceStep {
  step: string; status: 'running' | 'ok' | 'error' | 'skipped'; detail?: string | null; tool?: string | null
  duration_ms?: number | null; data?: Record<string, unknown> | null
}
export interface Provenance {
  pipeline_version: string; mode: 'demo' | 'live'; models: Record<string, string>; prompt_versions: Record<string, string>
  tool_calls: { tool: string; status: string; duration_ms: number }[]; record_sha256?: string | null
  started_at?: string | null; finished_at?: string | null
}
export interface AnalysisResult {
  id: string; status: 'queued' | 'running' | 'completed' | 'failed'; created_at: string; question: string; domain: Domain; intent?: string | null
  mode: 'demo' | 'live'; answer: string; confidence?: number | null; metrics: Record<string, Measurement>; observations: Observation[]
  assumptions: string[]; limitations: string[]; sources: Source[]; needs_review: boolean; review_reasons: string[]
  review_status: 'not_required' | 'pending' | 'approved' | 'rejected' | 'corrected'; next_actions: string[]
  clarification?: string | null; error?: string | null; trace: TraceStep[]; provenance?: Provenance | null
}

export interface Scene {
  id: string; domain: Domain; title: string; location: string; description: string; sensor?: string
  before_at?: string; after_at?: string; width: number; height: number; crs?: string | null; gsd_m?: number | null
  bounds_wgs84?: number[] | null
  suggested: string[]
  assets: { after: string; before?: string }
  upload?: boolean
}
export interface AssetInfo {
  id: string; filename?: string | null; width: number; height: number; bands: string[]; georeferenced: boolean
  crs?: string | null; gsd_m?: number | null; captured_at?: string | null; bounds_wgs84?: number[] | null
}

export interface StreamHandlers { onStep: (s: TraceStep) => void; onToolStart: (tool: string) => void }

export interface Backend {
  kind: 'static' | 'live'
  label: string
  canUpload: boolean
  listScenes(): Promise<Scene[]>
  imageUrl(scene: Scene, which: 'after' | 'before'): Promise<string>
  layersUrl?(scene: Scene): string
  analyze(scene: Scene, question: string, h: StreamHandlers, signal?: AbortSignal): Promise<AnalysisResult>
  upload(file: File): Promise<AssetInfo>
  review(result: AnalysisResult, decision: 'approved' | 'rejected' | 'corrected', comment?: string): Promise<AnalysisResult['review_status']>
}
