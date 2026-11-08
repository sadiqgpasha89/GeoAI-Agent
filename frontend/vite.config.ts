/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base './' keeps the build relocatable (GitHub Pages project sites live under /<repo>/).
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { sourcemap: true, chunkSizeWarningLimit: 700 },
  server: { port: 5173 },
  test: { environment: 'jsdom', include: ['src/**/*.test.{ts,tsx}'] },
})
