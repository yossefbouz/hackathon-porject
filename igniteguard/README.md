# IgniteGuard

Prototype codebase for the IgniteGuard fire-prevention MVP. Detailed setup and
usage instructions will follow in later phases.

## Frontend wildfire risk map

The `igniteguard/frontend` folder contains a Vite + React single-page
application that visualises the Cyprus wildfire spread risk index on a Leaflet
map. Data is sourced from `igniteguard/data/cyprus_risk_index.csv` and rendered
with a red (high spread potential) to green (low spread potential) colour scale.

```bash
cd igniteguard/frontend
npm install
npm run dev
```

The development server will start on http://localhost:5173/ and automatically
open your browser.
