import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'fs'
import { resolve } from 'path'

// 读取 package.json 获取版本号
const packageJson = JSON.parse(
  readFileSync(resolve(__dirname, 'package.json'), 'utf-8')
)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // 定义全局常量，在构建时注入
  define: {
    'import.meta.env.VITE_APP_VERSION': JSON.stringify(packageJson.version),
    'import.meta.env.VITE_BUILD_TIME': JSON.stringify(
      new Date().toISOString().split('T')[0]
    ),
  },
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
