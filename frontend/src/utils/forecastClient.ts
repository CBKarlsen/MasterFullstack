/**
 * Client-side clogging trajectory forecasting.
 *
 * Mirrors app/forecast.py so the forecast updates instantly when the user
 * changes sigma — no server round-trip required.
 */
import type { AnalysisPoint } from "../components/BatchAnalysisCards";

// ── Types ──────────────────────────────────────────────────────────────────

export interface FitResult {
  r2: number;
  params: Record<string, number>;
  eta_seconds: number | null;
}

export interface ForecastData {
  onset_time: number;
  detection_threshold: number;
  critical_threshold: number;
  critical_multiplier: number;
  fits: Record<string, FitResult>;
  best_fit: string;
  consensus_eta: number | null;
  curve_data: Record<string, Array<{ time: number; value: number }>>;
  post_onset_points: number;
}

// Internal type before converting relative ETA to absolute timestamp
interface RawFit {
  r2: number;
  params: Record<string, number>;
  etaRel: number | null; // seconds after onset; null = no crossing projected
}

// ── Constants ─────────────────────────────────────────────────────────────

const MIN_FORECAST_POINTS = 20;
const DEFAULT_CRITICAL_MULTIPLIER = 2.0;
const CURVE_POINTS_COUNT = 100;
const MIN_GROWTH_RATE = 1e-10;
const MAX_ETA_SECONDS = 30 * 24 * 3600; // treat anything beyond 30 days as no crossing

// ── Math helpers ───────────────────────────────────────────────────────────

function rSquared(actual: number[], predicted: number[]): number {
  const mean = actual.reduce((a, b) => a + b, 0) / actual.length;
  const ssTot = actual.reduce((a, y) => a + (y - mean) ** 2, 0);
  const ssRes = actual.reduce((a, y, i) => a + (y - predicted[i]) ** 2, 0);
  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}

function linfit(x: number[], y: number[]): { slope: number; intercept: number } {
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((a, xi, i) => a + xi * y[i], 0);
  const sumXX = x.reduce((a, xi) => a + xi * xi, 0);
  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-14) return { slope: 0, intercept: sumY / n };
  return {
    slope: (n * sumXY - sumX * sumY) / denom,
    intercept: (sumY - sumX * ((n * sumXY - sumX * sumY) / denom)) / n,
  };
}

// ── Model fits ─────────────────────────────────────────────────────────────

function fitLinear(tRel: number[], scores: number[], critical: number): RawFit {
  const { slope: a, intercept: b } = linfit(tRel, scores);
  const predicted = tRel.map((t) => a * t + b);
  const r2 = rSquared(scores, predicted);
  const etaRel = a > MIN_GROWTH_RATE ? (critical - b) / a : null;
  return { r2, params: { a, b }, etaRel };
}

function fitExponential(tRel: number[], scores: number[], critical: number): RawFit {
  const validIdx = scores.map((s, i) => (s > 0 ? i : -1)).filter((i) => i >= 0);
  if (validIdx.length < MIN_FORECAST_POINTS) return { r2: 0, params: {}, etaRel: null };
  const logS = validIdx.map((i) => Math.log(scores[i]));
  const tV = validIdx.map((i) => tRel[i]);
  const { slope: k, intercept: logS0 } = linfit(tV, logS);
  const s0 = Math.exp(logS0);
  const r2 = rSquared(logS, tV.map((t) => k * t + logS0));
  const etaRel = k > MIN_GROWTH_RATE && s0 > 0 ? Math.log(critical / s0) / k : null;
  return { r2, params: { k, s0 }, etaRel };
}

function fitPowerLaw(tRel: number[], scores: number[], critical: number): RawFit {
  const validIdx = scores.map((s, i) => (s > 0 && tRel[i] > 0 ? i : -1)).filter((i) => i >= 0);
  if (validIdx.length < MIN_FORECAST_POINTS) return { r2: 0, params: {}, etaRel: null };
  const logS = validIdx.map((i) => Math.log(scores[i]));
  const logT = validIdx.map((i) => Math.log(tRel[i]));
  const { slope: n, intercept: logA } = linfit(logT, logS);
  const a = Math.exp(logA);
  const r2 = rSquared(logS, logT.map((t) => n * t + logA));
  const etaRel = n > MIN_GROWTH_RATE && a > 0 ? (critical / a) ** (1 / n) : null;
  return { r2, params: { a, n }, etaRel };
}

function toAbsoluteEta(etaRel: number | null, onsetTime: number): number | null {
  if (etaRel === null || etaRel <= 0 || etaRel >= MAX_ETA_SECONDS) return null;
  return onsetTime + etaRel;
}

// ── Curve generation ───────────────────────────────────────────────────────

function evalModel(model: string, params: Record<string, number>, t: number): number {
  if (model === "linear") return params.a * t + params.b;
  if (model === "exponential") return params.s0 * Math.exp(params.k * t);
  if (model === "power_law") return params.a * Math.max(t, 1e-9) ** params.n;
  return 0;
}

function generateCurvePoints(
  model: string,
  params: Record<string, number>,
  tRelMax: number,
  onsetTime: number,
): Array<{ time: number; value: number }> {
  if (Object.keys(params).length === 0) return [];
  const step = (tRelMax * 2.0) / CURVE_POINTS_COUNT;
  return Array.from({ length: CURVE_POINTS_COUNT }, (_, i) => {
    const tRel = i * step;
    const value = evalModel(model, params, tRel);
    if (!isFinite(value) || value < 0 || value > 10) return null;
    return { time: onsetTime + tRel, value };
  }).filter((p): p is { time: number; value: number } => p !== null);
}

// ── Main export ────────────────────────────────────────────────────────────

export function computeForecast(
  timeseries: AnalysisPoint[],
  fftThreshold: number,
  criticalMultiplier: number = DEFAULT_CRITICAL_MULTIPLIER,
): ForecastData | null {
  const onset = timeseries.find(
    (p) => p.phase === "analysis" && p.composite_score > fftThreshold,
  );
  if (!onset) return null;

  const onsetTime = onset.time;
  const critical = fftThreshold * criticalMultiplier;
  const postOnset = timeseries.filter((p) => p.phase === "analysis" && p.time >= onsetTime);
  if (postOnset.length < MIN_FORECAST_POINTS) return null;

  const tRel = postOnset.map((p) => p.time - onsetTime);
  const scores = postOnset.map((p) => p.composite_score);
  const tRelMax = tRel[tRel.length - 1];

  const rawFits: Record<string, RawFit> = {
    linear: fitLinear(tRel, scores, critical),
    exponential: fitExponential(tRel, scores, critical),
    power_law: fitPowerLaw(tRel, scores, critical),
  };

  const fits: Record<string, FitResult> = Object.fromEntries(
    Object.entries(rawFits).map(([name, raw]) => [
      name,
      { r2: raw.r2, params: raw.params, eta_seconds: toAbsoluteEta(raw.etaRel, onsetTime) },
    ]),
  );

  const validEtas = Object.values(fits)
    .map((f) => f.eta_seconds)
    .filter((eta): eta is number => eta !== null)
    .sort((a, b) => a - b);
  const consensusEta = validEtas.length > 0 ? validEtas[Math.floor(validEtas.length / 2)] : null;

  const bestFit = Object.entries(fits).reduce<[string, FitResult]>(
    (best, entry) => (entry[1].r2 > best[1].r2 ? entry : best),
    Object.entries(fits)[0],
  )[0];

  return {
    onset_time: onsetTime,
    detection_threshold: fftThreshold,
    critical_threshold: critical,
    critical_multiplier: criticalMultiplier,
    fits,
    best_fit: bestFit,
    consensus_eta: consensusEta,
    post_onset_points: postOnset.length,
    curve_data: Object.fromEntries(
      Object.entries(fits).map(([name, fit]) => [
        name,
        generateCurvePoints(name, fit.params, tRelMax, onsetTime),
      ]),
    ),
  };
}
