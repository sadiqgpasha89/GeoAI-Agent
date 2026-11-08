import { describe, expect, it } from 'vitest'
import { matchQuestion } from './api'
import { formatMeasure, polygonPoints, toCSV, toGeoJSON } from './lib'
import type { AnalysisResult, Observation } from './types'

const Q = [
  { id: '1', text: 'Describe this scene in detail' },
  { id: '2', text: 'How many buildings are flooded?' },
  { id: '3', text: 'Is the whole road flooded?' },
]
const obs = (o: Partial<Observation>): Observation => ({ id: 'o1', label: 'flooded_building', kind: 'observed', confidence: 0.9, attributes: {}, evidence: [], ...o })

describe('matchQuestion', () => {
  it('matches exact and paraphrased questions', () => {
    expect(matchQuestion('How many buildings are flooded?', Q)?.id).toBe('2')
    expect(matchQuestion('number of flooded buildings', Q)?.id).toBe('2')
    expect(matchQuestion('is the road flooded', Q)?.id).toBe('3')
  })
  it('returns null for unrelated text', () => {
    expect(matchQuestion('what is the capital of France', Q)).toBeNull()
  })
})

describe('exports', () => {
  const res = { id: 'a1', question: 'q', observations: [obs({ geometry: { type: 'Point', coordinates: [1, 2] } }), obs({ id: 'o2', label: '=SUM(A1)', geometry: null })] } as unknown as AnalysisResult
  it('keeps null geometry for non-georeferenced observations', () => {
    const gj = toGeoJSON(res)
    expect(gj.features[0].geometry).not.toBeNull()
    expect(gj.features[1].geometry).toBeNull()
  })
  it('neutralises CSV formula injection', () => {
    expect(toCSV(res)).toContain(`"'=SUM(A1)"`)
  })
})

describe('helpers', () => {
  it('builds polygon points from polygon or bbox', () => {
    expect(polygonPoints(obs({ pixel_polygon: [[0, 0], [1, 0], [1, 1]] }))).toBe('0,0 1,0 1,1')
    expect(polygonPoints(obs({ bbox_px: [1, 2, 3, 4] }))).toBe('1,2 3,2 3,4 1,4')
    expect(polygonPoints(obs({}))).toBeNull()
  })
  it('formats measurements', () => {
    expect(formatMeasure({ value: 1234.5678, unit: 'ha' })).toMatch(/1,?234\.57 ha/)
    expect(formatMeasure({ value: null })).toBe('—')
  })
})
