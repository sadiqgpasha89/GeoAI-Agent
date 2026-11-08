import { useEffect, useRef, useState } from 'react'
import { testConnection, type LiveConfig } from '../api'

interface Props { open: boolean; initial: LiveConfig | null; onClose: () => void; onSave: (cfg: LiveConfig | null) => void }

export default function SettingsDialog({ open, initial, onClose, onSave }: Props) {
  const ref = useRef<HTMLDialogElement>(null)
  const [baseUrl, setBaseUrl] = useState(initial?.baseUrl ?? 'http://localhost:8000')
  const [apiKey, setApiKey] = useState(initial?.apiKey ?? '')
  const [status, setStatus] = useState<{ ok: boolean; msg: string } | null>(null)

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open])

  const test = async () => {
    setStatus({ ok: true, msg: 'Testing…' })
    try { setStatus({ ok: true, msg: await testConnection({ baseUrl, apiKey }) }) } catch (e) { setStatus({ ok: false, msg: (e as Error).message || 'Could not reach the server (CORS or network)' }) }
  }

  return (
    <dialog ref={ref} className="dialog" onClose={onClose} aria-labelledby="settings-title">
      <form method="dialog" onSubmit={(e) => { e.preventDefault(); onSave({ baseUrl, apiKey }) }}>
        <h2 id="settings-title">Connect to a TerraLens API</h2>
        <p className="muted">By default this site runs as a <b>static demo</b>: results are pre-computed by the real agent pipeline. Connect your own backend for free-form questions, uploads and live VLM analysis.</p>
        <label>API base URL<input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://geoai.example.com" inputMode="url" required /></label>
        <label>API key (optional)<input value={apiKey} onChange={(e) => setApiKey(e.target.value)} type="password" autoComplete="off" placeholder="X-API-Key" /></label>
        <p className="hint">The key stays in this browser's local storage and is sent only to the URL above. The backend must allow this origin via <code>GEOAI_CORS_ORIGINS</code>.</p>
        {status && <p className={status.ok ? 'ok-text' : 'error-text'} role="status">{status.msg}</p>}
        <div className="dialog-actions">
          <button type="button" className="ghost" onClick={test}>Test connection</button>
          <span className="spacer" />
          {initial && <button type="button" className="ghost danger" onClick={() => onSave(null)}>Back to static demo</button>}
          <button type="button" className="ghost" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary">Connect</button>
        </div>
      </form>
    </dialog>
  )
}
