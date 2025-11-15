import { useMemo } from "react";
import RiskMap from "./components/RiskMap";
import { useRiskDataset } from "./hooks/useRiskDataset";
import "./App.css";

const RISK_DATASET_URL = new URL(
  "../../data/cyprus_risk_index.csv",
  import.meta.url
).href;

function App() {
  const { records, loading, error } = useRiskDataset(RISK_DATASET_URL);

  const summary = useMemo(() => {
    if (!records.length) {
      return null;
    }

    const highestRisk = records.reduce((acc, current) =>
      current.riskIndex > acc.riskIndex ? current : acc
    );
    const lowestRisk = records.reduce((acc, current) =>
      current.riskIndex < acc.riskIndex ? current : acc
    );

    const averageRisk =
      records.reduce((acc, current) => acc + current.riskIndex, 0) /
      records.length;

    const averageTrees =
      records.reduce((acc, current) => acc + current.treeCoverPct, 0) /
      records.length;

    return {
      count: records.length,
      highestRisk,
      lowestRisk,
      averageRisk,
      averageTrees,
    };
  }, [records]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>IgniteGuard · Cyprus Wildfire Spread Potential</h1>
        <p>
          Colour-coded Leaflet map showing the current fire spread risk across
          Cyprus. Red markers highlight areas with intense winds and dense
          vegetation; green markers denote calmer zones with sparse vegetation.
        </p>
      </header>

      {loading && <p>Loading up-to-date risk observations…</p>}
      {error && (
        <p role="alert">Could not load the risk dataset: {error.message}</p>
      )}

      {summary && (
        <section className="dataset-summary" aria-label="Dataset summary">
          <article className="summary-card">
            <h3>Locations analysed</h3>
            <p>{summary.count} monitoring sites across Cyprus</p>
          </article>
          <article className="summary-card">
            <h3>Highest spread risk</h3>
            <p>
              {summary.highestRisk.name} · Risk index {summary.highestRisk.riskIndex.toFixed(
                2
              )}
            </p>
          </article>
          <article className="summary-card">
            <h3>Lowest spread risk</h3>
            <p>
              {summary.lowestRisk.name} · Risk index {summary.lowestRisk.riskIndex.toFixed(
                2
              )}
            </p>
          </article>
          <article className="summary-card">
            <h3>Mean conditions</h3>
            <p>
              Avg. risk index {summary.averageRisk.toFixed(2)} · Avg. tree cover {" "}
              {summary.averageTrees.toFixed(0)}%
            </p>
          </article>
        </section>
      )}

      <section className="map-wrapper" aria-label="Risk map">
        <RiskMap records={records} />
      </section>

      <div className="legend" aria-label="Risk colour legend">
        <span>Low</span>
        <span className="legend-gradient" aria-hidden="true" />
        <span>High</span>
      </div>
    </div>
  );
}

export default App;
