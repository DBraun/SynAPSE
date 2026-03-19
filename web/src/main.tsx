import { createRoot } from 'react-dom/client'
import './index.css'
import { App } from './components/App'
import { AppProvider } from './context/AppContext'
import { ThemeProvider } from './context/ThemeContext'

createRoot(document.getElementById('root')!).render(
  <ThemeProvider>
    <AppProvider>
      <App />
    </AppProvider>
  </ThemeProvider>,
)
