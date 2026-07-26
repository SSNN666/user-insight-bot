import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ask': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/tasks': 'http://localhost:8000',
      '/stats': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
      '/debug': 'http://localhost:8000',
      '/annotate': 'http://localhost:8000',
    },
  },
})
