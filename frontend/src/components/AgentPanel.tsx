import { useEffect, useRef, useState } from 'react'
import { pctText } from '../lib'
import type { AnalysisResult, TraceStep } from '../types'

export interface Msg { id: string; role: 'user' | 'agent'; text: string; result?: AnalysisResult; steps?: TraceStep[]; pending?: boolean; error?: boolean; tool?: string; example?: boolean }

interface Props { messages: Msg[]; suggestions: string[]; busy: boolean; placeholder: string; backendLabel: string; onAsk: (q: string) => void; onSelectResult: (r: AnalysisResult) => void }

const STEP_LABEL: Record<string, string> = {
  normalize: 'Validate inputs', classify: 'Classify intent', plan: 'Plan tools', validate: 'Validate results', review_gate: 'Review gate', synthesize: 'Synthesize answer',
  clarify: 'Ask to clarify', decline: 'Decline request',
}

export function StepList({ steps, running }: { steps: TraceStep[]; running?: string }) {
  return (
    <ol className="steps" aria-label="Agent steps">
      {steps.map((s, i) => (
        <li key={i} className={s.status}>
          <span className="step-dot" />
          <div>
            <b>{s.step === 'tool' ? s.tool : STEP_LABEL[s.step] ?? s.step}</b>
            {s.detail && <small>{s.detail}</small>}
          </div>
          {s.duration_ms != null && <em>{s.duration_ms} ms</em>}
        </li>
      ))}
      {running && <li className="running"><span className="step-dot" /><div><b>{running}</b><small>running…</small></div></li>}
    </ol>
  )
}

export default function AgentPanel({ messages, suggestions, busy, placeholder, backendLabel, onAsk, onSelectResult }: Props) {
  const [text, setText] = useState('')
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ block: 'end', behavior: 'smooth' }) }, [messages])
  const submit = () => { const q = text.trim(); if (!q || busy) return; setText(''); onAsk(q) }

  return (
    <aside className="agent-panel" aria-label="TerraLens agent">
      <div className="agent-header">
        <div className="agent-orb" aria-hidden>✦</div>
        <div><h2>TerraLens agent</h2><p><span className={`status-dot ${busy ? 'busy' : ''}`} /> {busy ? 'Investigating…' : 'Ready to investigate'}</p></div>
        <span className="mode-pill">{backendLabel}</span>
      </div>
      <div className="conversation" role="log" aria-live="polite">
        {messages.map((m) => m.role === 'user' ? (
          <div className="user-message" key={m.id}><div className="user-bubble">{m.text}{m.example && <small> · example</small>}</div></div>
        ) : (
          <div className="agent-message" key={m.id}>
            <div className="mini-orb" aria-hidden>✦</div>
            <div className="agent-body">
              {m.pending ? <StepList steps={m.steps ?? []} running={m.tool ?? 'Working'} /> : (
                <>
                  <p className={m.error ? 'error-text' : ''}>{m.text}</p>
                  {m.result && (
                    <div className="response-meta">
                      {m.result.status === 'completed' && m.result.observations.length > 0 && <span>✓ Grounded in {m.result.observations.length} evidence regions</span>}
                      {m.result.confidence != null && <b>{pctText(m.result.confidence)} confidence</b>}
                      {m.result.needs_review && <span className={`badge review ${m.result.review_status}`}>{m.result.review_status === 'pending' ? 'Needs analyst review' : `Review: ${m.result.review_status}`}</span>}
                      <button className="link" onClick={() => onSelectResult(m.result!)}>Show on image</button>
                    </div>
                  )}
                  {m.result && m.result.trace.length > 0 && (
                    <details className="trace"><summary>Agent steps ({m.result.trace.length})</summary><StepList steps={m.result.trace} /></details>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={end} />
      </div>
      <div className="agent-tools">
        <span>Suggested analysis</span>
        {suggestions.map((s) => <button key={s} className="tool-chip" disabled={busy} onClick={() => onAsk(s)}>{s}</button>)}
      </div>
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); submit() }}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder={placeholder} aria-label="Ask a question about the imagery" maxLength={1000} disabled={busy} />
        <button type="submit" aria-label="Send question" disabled={busy || !text.trim()}>↑</button>
      </form>
      <div className="agent-footer"><span>Enter to send</span><span><i className="green-dot" /> Evidence-grounded responses</span></div>
    </aside>
  )
}
