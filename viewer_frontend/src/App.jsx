import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getPredictions, getRunDetail, getRuns } from "./api";

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "NA";
  }
  return Number(value).toFixed(digits);
}

function metricDeltaClass(value) {
  if (value === null || value === undefined) return "";
  return value > 0 ? "metric-positive" : value < 0 ? "metric-negative" : "";
}

function SummaryCard({ label, value, hint, className = "" }) {
  return (
    <div className={`summary-card ${className}`}>
      <div className="summary-label">{label}</div>
      <div className="summary-value">{value}</div>
      {hint ? <div className="summary-hint">{hint}</div> : null}
    </div>
  );
}

function RunsTable({ runs, selectedRunId, onSelectRun }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Runs</h2>
        <span>{runs.length} indexed</span>
      </div>
      <div className="table-wrap">
        <table className="runs-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Mode</th>
              <th>Test MAE</th>
              <th>Baseline MAE</th>
              <th>Tickers</th>
              <th>Predictions</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr
                key={run.id}
                className={run.id === selectedRunId ? "selected-row" : ""}
                onClick={() => onSelectRun(run.id)}
              >
                <td>
                  <div className="run-name">{run.name}</div>
                  <div className="run-subtitle">
                    {run.model_type || "unknown"}
                    {run.optimizer ? ` / ${run.optimizer}` : ""}
                  </div>
                </td>
                <td>{run.has_sector_breakdown ? `per-sector (${run.sector_count})` : "single"}</td>
                <td>{formatNumber(run.test_mae)}</td>
                <td>{formatNumber(run.baseline_test_mae)}</td>
                <td>{run.ticker_count ?? "NA"}</td>
                <td>{run.prediction_count ?? "NA"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TrainingHistoryChart({ history }) {
  if (!history?.length) {
    return (
      <div className="chart-card">
        <div className="panel-header">
          <h3>Training History</h3>
        </div>
        <div className="loading">No epoch history was saved for this run.</div>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <div className="panel-header">
        <h3>Training History</h3>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={history}>
          <CartesianGrid strokeDasharray="3 3" stroke="#28303a" />
          <XAxis dataKey="epoch" stroke="#93a1b2" />
          <YAxis stroke="#93a1b2" />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="train_mae" stroke="#ff7a59" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="val_mae" stroke="#2ec4b6" dot={false} strokeWidth={2} />
          {history.some((row) => row.lr !== null && row.lr !== undefined) ? (
            <Line type="monotone" dataKey="lr" stroke="#8ecae6" dot={false} strokeWidth={1.5} />
          ) : null}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function PredictionScatter({ rows }) {
  if (!rows?.length) {
    return (
      <div className="chart-card">
        <div className="panel-header">
          <h3>Prediction Scatter</h3>
        </div>
        <div className="loading">No prediction summary is available for this run.</div>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <div className="panel-header">
        <h3>Prediction Scatter</h3>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 16, right: 16, bottom: 16, left: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#28303a" />
          <XAxis type="number" dataKey="actual" name="Actual" stroke="#93a1b2" />
          <YAxis type="number" dataKey="prediction" name="Prediction" stroke="#93a1b2" />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter name="Predictions" data={rows} fill="#ffb703" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function TickerTable({ rows }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Ticker Error Summary</h3>
      </div>
      <div className="table-wrap">
        <table className="runs-table compact-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Samples</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>Mean Pred</th>
              <th>Mean Actual</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 30).map((row) => (
              <tr key={row.ticker}>
                <td>{row.ticker}</td>
                <td>{row.samples}</td>
                <td>{formatNumber(row.mae)}</td>
                <td>{formatNumber(row.rmse)}</td>
                <td>{formatNumber(row.mean_prediction)}</td>
                <td>{formatNumber(row.mean_actual)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SectorTable({ rows }) {
  if (!rows?.length) return null;
  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Sector Results</h3>
      </div>
      <div className="table-wrap">
        <table className="runs-table compact-table">
          <thead>
            <tr>
              <th>Sector</th>
              <th>Test MAE</th>
              <th>Test RMSE</th>
              <th>Train</th>
              <th>Val</th>
              <th>Test</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.sector_bucket}-${row.run_id || "root"}`}>
                <td>{row.sector_bucket}</td>
                <td>{formatNumber(row.test_mae)}</td>
                <td>{formatNumber(row.test_rmse)}</td>
                <td>{row.num_train ?? "NA"}</td>
                <td>{row.num_val ?? "NA"}</td>
                <td>{row.num_test ?? "NA"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PredictionExplorer({ selectedRunId }) {
  const [ticker, setTicker] = useState("");
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedRunId) return;
    let active = true;
    setLoading(true);
    getPredictions(selectedRunId, {
      ticker: ticker || undefined,
      sortBy: "abs_error",
      descending: true,
      limit: 200,
    })
      .then((payload) => {
        if (active) {
          setRows(payload.rows);
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [selectedRunId, ticker]);

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Prediction Explorer</h3>
        <div className="filter-row">
          <input
            value={ticker}
            onChange={(event) => setTicker(event.target.value.toUpperCase())}
            placeholder="Filter ticker"
          />
        </div>
      </div>
      {loading ? <div className="loading">Loading predictions…</div> : null}
      <div className="table-wrap">
        <table className="runs-table compact-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Published</th>
              <th>Pred</th>
              <th>Actual</th>
              <th>Abs Err</th>
              <th>Prior EPS</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.sample_id}-${row.ticker}`}>
                <td>{row.ticker}</td>
                <td>{String(row.target_published_date).slice(0, 10)}</td>
                <td>{formatNumber(row.prediction)}</td>
                <td>{formatNumber(row.actual)}</td>
                <td>{formatNumber(row.abs_error)}</td>
                <td>{formatNumber(row.last_prior_eps)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function App() {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedRun, setSelectedRun] = useState(null);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let active = true;
    getRuns()
      .then((payload) => {
        if (!active) return;
        setLoadError("");
        setRuns(payload.runs);
        if (payload.runs.length > 0) {
          setSelectedRunId(payload.runs[0].id);
        }
      })
      .catch((error) => {
        if (!active) return;
        setLoadError(String(error?.message || error));
      })
      .finally(() => {
        if (active) setLoadingRuns(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    let active = true;
    getRunDetail(selectedRunId)
      .then((payload) => {
        if (active) {
          setLoadError("");
          setSelectedRun(payload);
        }
      })
      .catch((error) => {
        if (active) {
          setLoadError(String(error?.message || error));
          setSelectedRun(null);
        }
      });
    return () => {
      active = false;
    };
  }, [selectedRunId]);

  const comparison = useMemo(() => {
    if (!selectedRun?.summary) return null;
    const run = selectedRun.summary;
    const delta =
      run.baseline_test_mae !== null && run.baseline_test_mae !== undefined &&
      run.test_mae !== null && run.test_mae !== undefined
        ? run.baseline_test_mae - run.test_mae
        : null;
    return {
      delta,
      pct: delta !== null && run.baseline_test_mae ? (delta / run.baseline_test_mae) * 100 : null,
    };
  }, [selectedRun]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <div className="eyebrow">EPS Estimation</div>
          <h1>Run Viewer</h1>
        </div>
        <div className="header-note">Interactive inspection for training metrics, sector breakdowns, curves, and prediction error structure.</div>
      </header>

      {loadingRuns ? <div className="loading">Loading runs…</div> : null}
      {loadError ? <div className="error-banner">Failed to load run data: {loadError}</div> : null}
      {!loadingRuns && !loadError && runs.length === 0 ? (
        <div className="error-banner">No runs were returned by the API. Check that the backend can see `artifacts/*_training`.</div>
      ) : null}

      <div className="layout">
        <div className="layout-left">
          <RunsTable runs={runs} selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} />
        </div>
        <div className="layout-right">
          {selectedRun ? (
            <>
              <div className="summary-grid">
                <SummaryCard label="Run" value={selectedRun.summary.name} hint={`${selectedRun.summary.model_type || "unknown"}${selectedRun.summary.optimizer ? ` / ${selectedRun.summary.optimizer}` : ""}`} />
                <SummaryCard label="Test MAE" value={formatNumber(selectedRun.summary.test_mae)} />
                <SummaryCard label="Test RMSE" value={formatNumber(selectedRun.summary.test_rmse)} />
                <SummaryCard
                  label={selectedRun.summary.baseline_name ? `Vs ${selectedRun.summary.baseline_name}` : "Vs Baseline"}
                  value={comparison?.delta !== null ? `${formatNumber(comparison?.delta)} MAE` : "NA"}
                  hint={comparison?.pct !== null ? `${formatNumber(comparison?.pct, 1)}% better` : undefined}
                  className={metricDeltaClass(comparison?.delta)}
                />
                <SummaryCard label="Tickers" value={selectedRun.summary.ticker_count} />
                <SummaryCard label="Predictions" value={selectedRun.summary.prediction_count} />
              </div>

              <TrainingHistoryChart history={selectedRun.history} />
              <SectorTable rows={selectedRun.sector_summary} />
              <PredictionScatter rows={selectedRun.ticker_summary.slice(0, 80).map((row) => ({
                actual: row.mean_actual,
                prediction: row.mean_prediction,
                ticker: row.ticker,
              }))} />
              <TickerTable rows={selectedRun.ticker_summary} />
              <PredictionExplorer selectedRunId={selectedRunId} />
            </>
          ) : (
            <div className="loading">Select a run to inspect it.</div>
          )}
        </div>
      </div>
    </div>
  );
}
