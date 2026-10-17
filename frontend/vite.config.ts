import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/recommend': 'http://127.0.0.1:8000',
      '/movies': 'http://127.0.0.1:8000',
      '/genres': 'http://127.0.0.1:8000',
      '/users': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/feedback': 'http://127.0.0.1:8000',
    }
  }
})
