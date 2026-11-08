import type { AnalysisResult, AssetInfo, Backend, Scene, StreamHandlers, TraceStep } from './types'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const STOP = new Set('the a an of is are in on at to for and or with this that these those it be by as from any all me you your what which how'.split(' '))
const tokens = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((t) => t.length > 2 && !STOP.has(t))

/** Jaccard similarity on content words — used to map free text onto the pre-computed demo questions. */
export function matchQuestion<T extends { id: string; text: string }>(text: string, candidates: T[], min = 0.34): T | null {
  const a = new Set(tokens(text))
  let best: T | null = null
  let bestScore = 0
  for (const c of candidates) {
    if (c.text.toLowerCase() === text.trim().toLowerCase()) return c
    const b = new Set(tokens(c.text))
    const inter = [...a].filter((t) => b.has(t)).length
    const union = new Set([...a, ...b]).size || 1
    const score = inter / union
    if (score > bestScore) { bestScore = score; best = c }
  }
  return bestScore >= min ? best : null
}

interface ManifestScene extends Omit<Scene, 'suggested' | 'assets'> {
  images: { after: string; before: string }
  layers: string
  questions: { id: string; text: string; result: string }[]
}

/** Static demo backend: results were produced offline by the real agent pipeline; the UI replays the recorded trace. */
export function createStaticBackend(base = import.meta.env.BASE_URL): Backend {
  const root = `${base}demo/`
  const reviews = new Map<string, AnalysisResult['review_status']>()
  type Manifest = { scenes: ManifestScene[] }
  let manifest: Promise<Manifest> | null = null
  const load = (): Promise<Manifest> => (manifest ??= fetch(`${root}manifest.json`).then((r) => {
    if (!r.ok) throw new Error(`Could not load demo data (${r.status})`)
    return r.json() as Promise<Manifest>
  }))
  const find = async (id: string) => (await load()).scenes.find((s) => s.id === id)!

  return {
    kind: 'static',
    label: 'Static demo',
    canUpload: false,
    async listScenes() {
      return (await load()).scenes.map((s) => ({
        ...s, suggested: s.questions.slice(0, 5).map((q) => q.text), assets: { after: s.images.after, before: s.images.before },
      }))
    },
    async imageUrl(scene, which) { return root + (which === 'after' ? scene.assets.after : scene.assets.before!) },
    layersUrl: (scene) => `${root}scenes/${scene.id}/layers.geojson`,
    async analyze(scene, question, h: StreamHandlers) {
      const ms = await find(scene.id)
      const hit = matchQuestion(question, ms.questions)
      await sleep(250)
      if (!hit) {
        const empty = {
          id: `static_${Date.now()}`, status: 'completed', created_at: new Date().toISOString(), question, domain: scene.domain, mode: 'demo',
          answer: 'The static demo can only answer the pre-computed questions below. Connect the live API (Settings) for free-form questions and your own imagery.',
          metrics: {}, observations: [], assumptions: [], limitations: ['Static demo: results are pre-computed.'], sources: [], needs_review: false,
          review_reasons: [], review_status: 'not_required', next_actions: ms.questions.map((q) => q.text), trace: [],
        } as AnalysisResult
        return empty
      }
      const res: AnalysisResult = await fetch(root + hit.result).then((r) => r.json())
      res.question = question
      for (const step of res.trace) {
        if (step.tool) h.onToolStart(step.tool)
        await sleep(step.step === 'tool' ? 520 : 300)
        h.onStep(step)
      }
      res.review_status = reviews.get(res.id) ?? res.review_status
      return res
    },
    async upload() { throw new Error('Uploads need the live API. Open Settings to connect one.') },
    async review(result, decision) { reviews.set(result.id, decision); return decision },
  }
}

export interface LiveConfig { baseUrl: string; apiKey: string }

export function createLiveBackend(cfg: LiveConfig): Backend {
  const api = `${cfg.baseUrl.replace(/\/$/, '')}/api/v1`
  const headers = (extra: Record<string, string> = {}) => ({ ...(cfg.apiKey ? { 'X-API-Key': cfg.apiKey } : {}), ...extra })
  const json = async <T>(path: string, init: RequestInit = {}): Promise<T> => {
    const r = await fetch(api + path, { ...init, headers: headers(init.headers as Record<string, string>) })
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`
      try { const b = await r.json(); msg = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail) } catch { /* keep default */ }
      throw new Error(msg)
    }
    return r.json()
  }
  const blobCache = new Map<string, string>()

  return {
    kind: 'live',
    label: 'Live API',
    canUpload: true,
    async listScenes() {
      const raw = await json<(Omit<Scene, 'suggested' | 'assets'> & { suggested_questions: string[]; assets: Scene['assets'] })[]>('/samples')
      return raw.map(({ suggested_questions, ...s }) => ({ ...s, suggested: suggested_questions }))
    },
    async imageUrl(scene, which) {
      const id = which === 'after' ? scene.assets.after : scene.assets.before
      if (!id) throw new Error('No earlier image for this scene')
      const hit = blobCache.get(id)
      if (hit) return hit
      const r = await fetch(`${api}/assets/${encodeURIComponent(id)}/preview`, { headers: headers() })
      if (!r.ok) throw new Error(`Image failed to load (${r.status})`)
      const url = URL.createObjectURL(await r.blob())
      blobCache.set(id, url)
      return url
    },
    async analyze(scene, question, h, signal) {
      const body = JSON.stringify({ question, domain: scene.domain, asset_id: scene.assets.after, before_asset_id: scene.assets.before ?? null, mode: 'auto' })
      const r = await fetch(`${api}/analyses/stream`, { method: 'POST', headers: headers({ 'Content-Type': 'application/json' }), body, signal })
      if (!r.ok || !r.body) throw new Error((await r.json().catch(() => ({ detail: r.statusText }))).detail ?? 'Analysis failed')
      const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
      let buf = ''
      let result: AnalysisResult | null = null
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += value
        let idx
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx); buf = buf.slice(idx + 2)
          const data = block.split('\n').find((l) => l.startsWith('data: '))
          if (!data) continue
          const ev = JSON.parse(data.slice(6))
          if (ev.event === 'step') h.onStep(ev.step as TraceStep)
          else if (ev.event === 'tool_start') h.onToolStart(ev.tool)
          else if (ev.event === 'result') result = ev.result
          else if (ev.event === 'error') throw new Error(ev.message)
        }
      }
      if (!result) throw new Error('The stream ended without a result')
      if (result.status === 'failed') throw new Error(result.error ?? 'Analysis failed')
      return result
    },
    async upload(file) {
      const fd = new FormData()
      fd.append('file', file)
      return json<AssetInfo>('/assets', { method: 'POST', body: fd })
    },
    async review(result, decision, comment) {
      await json(`/analyses/${result.id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision, comment, reviewer: 'analyst' }) })
      return decision
    },
  }
}

export async function testConnection(cfg: LiveConfig): Promise<string> {
  const r = await fetch(`${cfg.baseUrl.replace(/\/$/, '')}/api/v1/meta`, { headers: cfg.apiKey ? { 'X-API-Key': cfg.apiKey } : {} })
  if (!r.ok) throw new Error(r.status === 401 ? 'Invalid API key' : `Server responded ${r.status}`)
  const m = await r.json()
  return `Connected · v${m.version} · ${m.default_mode} engine${m.live_available ? ' + live models' : ''}`
}
