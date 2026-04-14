/**
 * Stores per-file sigma comparison data in localStorage and computes
 * forecasts at σ=3/4/5 without a server round-trip.
 */
import { computeForecast } from "./forecastClient";
import type { AnalysisPoint } from "../components/BatchAnalysisCards";

// ── Types ──────────────────────────────────────────────────────────────────

interface CalibrationStats {
  composite_mean: number;
  composite_std: number;
}

interface BatchResultForComputation {
  timeseries: AnalysisPoint[];
  calibration: CalibrationStats;
}

export interface SigmaForecastResult {
  onsetTime: number | null;
  consensusEta: number | null;
}

export interface SigmaRecord {
  filename: string;
  actualCloggingTime: number | null;
  forecasts: Record<number, SigmaForecastResult>;
}

export type SigmaStore = Record<string, SigmaRecord>;

export const SIGMA_VALUES = [3, 4, 5] as const;
export type SigmaValue = (typeof SIGMA_VALUES)[number];

const STORAGE_KEY = "sigma_comparison_data";

// ── Computation helpers ────────────────────────────────────────────────────

function computeThreshold(calibration: CalibrationStats, sigma: number): number {
  const { composite_mean, composite_std } = calibration;
  return composite_std > 0
    ? composite_mean + sigma * composite_std
    : Math.max(0.05, composite_mean * 5.0);
}

/**
 * First time the causal rolling mean of composite_score exceeds threshold.
 * Mirrors the logic in Batchanalysis.tsx `thresholdCrossings.composite`.
 */
function findCompositeCrossing(
  points: AnalysisPoint[],
  threshold: number,
  windowSec: number,
): number | null {
  let wStart = 0;
  for (let i = 0; i < points.length; i++) {
    while (points[wStart].time < points[i].time - windowSec) wStart++;
    const window = points.slice(wStart, i + 1).map((p) => p.composite_score);
    window.sort((a, b) => a - b);
    const mid = Math.floor(window.length / 2);
    const median = window.length % 2 === 1 ? window[mid] : (window[mid - 1] + window[mid]) / 2;
    if (median > threshold) return points[i].time;
  }
  return null;
}

/** Compute onset + consensus ETA for σ=3, 4, 5 in one pass (client-side only). */
export function computeSigmaForecasts(
  result: BatchResultForComputation,
  smoothingWindowSec: number,
): Record<SigmaValue, SigmaForecastResult> {
  const analysisPoints = result.timeseries.filter((p) => p.phase === "analysis");
  return Object.fromEntries(
    SIGMA_VALUES.map((sigma) => {
      const threshold = computeThreshold(result.calibration, sigma);
      const onsetTime = findCompositeCrossing(analysisPoints, threshold, smoothingWindowSec);
      const forecast = computeForecast(result.timeseries, threshold, {
        onsetTime: onsetTime ?? undefined,
        smoothingWindowSec,
      });
      return [sigma, { onsetTime, consensusEta: forecast?.consensus_eta ?? null }];
    }),
  ) as Record<SigmaValue, SigmaForecastResult>;
}

// ── localStorage CRUD ──────────────────────────────────────────────────────

export function loadStore(): SigmaStore {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as SigmaStore) : {};
  } catch {
    return {};
  }
}

function persistStore(store: SigmaStore): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

export function saveRecord(
  filename: string,
  forecasts: Record<SigmaValue, SigmaForecastResult>,
): void {
  const store = loadStore();
  store[filename] = {
    filename,
    // Preserve existing actual time if already annotated
    actualCloggingTime: store[filename]?.actualCloggingTime ?? null,
    forecasts,
  };
  persistStore(store);
}

export function setActualTime(filename: string, seconds: number): void {
  const store = loadStore();
  if (!store[filename]) return;
  store[filename].actualCloggingTime = seconds;
  persistStore(store);
}

export function clearStore(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function removeRecord(filename: string): void {
  const store = loadStore();
  delete store[filename];
  persistStore(store);
}
