import { useState } from "react";
import axios from "axios";
import type { SimulationData, ModelMetadata } from "../types";

interface Props {
  currentData: SimulationData | null;
  models: ModelMetadata[];
  isCalibrating: boolean;
  calibrationProgress: number;
  onToggle: (name: string, enabled: boolean) => void;
}

function probStyle(prob: number) {
  if (prob < 0.2) return { bar: "#22c55e", bg: "#f0fdf4", text: "#15803d", label: "Healthy" };
  if (prob < 0.7) return { bar: "#f59e0b", bg: "#fffbeb", text: "#d97706", label: "Warning" };
  return { bar: "#ef4444", bg: "#fef2f2", text: "#dc2626", label: "Critical" };
}

// ---------------------------------------------------------------------------
// Toggle switch (pill style)
// ---------------------------------------------------------------------------
function ToggleSwitch({
  enabled,
  loading,
  onClick,
}: {
  enabled: boolean;
  loading: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      aria-label={enabled ? "Disable model" : "Enable model"}
      style={{
        width: "34px",
        height: "18px",
        borderRadius: "9px",
        border: "none",
        backgroundColor: enabled ? "#16a34a" : "#d1d5db",
        cursor: loading ? "wait" : "pointer",
        position: "relative",
        transition: "background-color 0.2s",
        flexShrink: 0,
      }}
    >
      <span
        style={{
          position: "absolute",
          top: "2px",
          left: enabled ? "18px" : "2px",
          width: "14px",
          height: "14px",
          borderRadius: "50%",
          backgroundColor: "#fff",
          transition: "left 0.2s",
          boxShadow: "0 1px 2px rgba(0,0,0,0.15)",
        }}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Section header label
// ---------------------------------------------------------------------------
function SectionLabel({ children }: { children: string }) {
  return (
    <div
      style={{
        fontSize: "11px",
        fontWeight: 600,
        color: "#9ca3af",
        textTransform: "uppercase",
        letterSpacing: "0.06em",
        marginBottom: "10px",
      }}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function MLLivePanel({
  currentData,
  models,
  isCalibrating,
  calibrationProgress,
  onToggle,
}: Props) {
  const [toggling, setToggling] = useState<string | null>(null);

  const handleToggle = async (name: string, currentlyEnabled: boolean) => {
    setToggling(name);
    try {
      await axios.put(
        `/api/models/${encodeURIComponent(name)}/enable?enabled=${!currentlyEnabled}`,
      );
      onToggle(name, !currentlyEnabled);
    } catch {
      /* ignore — UI reverts on next poll */
    } finally {
      setToggling(null);
    }
  };

  const ensemble = currentData?.ensemble_probability;
  const hasData = !isCalibrating && ensemble != null;
  const ec = hasData ? probStyle(ensemble as number) : null;
  const preds = currentData?.models ?? {};
  const activeModels = models.filter((m) => m.enabled);

  // ── shared model toggles section ──────────────────────────────────────────
  const modelToggles = models.length > 0 && (
    <div style={{ borderTop: "1px solid #e5e7eb", padding: "12px 16px" }}>
      <SectionLabel>Active Models</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        {models.map((model) => (
          <div
            key={model.name}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
          >
            <span style={{ fontSize: "12px", color: model.enabled ? "#374151" : "#9ca3af" }}>
              {model.name}
            </span>
            <ToggleSwitch
              enabled={model.enabled}
              loading={toggling === model.name}
              onClick={() => handleToggle(model.name, model.enabled)}
            />
          </div>
        ))}
      </div>
    </div>
  );

  // ── calibrating state ─────────────────────────────────────────────────────
  if (isCalibrating) {
    return (
      <div
        style={{
          backgroundColor: "#fff",
          borderRadius: "8px",
          border: "1px solid #e5e7eb",
          overflow: "hidden",
        }}
      >
        <div style={{ padding: "16px" }}>
          <SectionLabel>ML Predictions</SectionLabel>
          <div style={{ textAlign: "center", padding: "8px 0 4px" }}>
            <div style={{ fontSize: "13px", color: "#6b7280", marginBottom: "10px" }}>
              Calibrating baseline…
            </div>
            <div
              style={{
                height: "6px",
                backgroundColor: "#e5e7eb",
                borderRadius: "3px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${calibrationProgress * 100}%`,
                  height: "100%",
                  backgroundColor: "#3b82f6",
                  transition: "width 0.3s",
                }}
              />
            </div>
            <div style={{ fontSize: "11px", color: "#9ca3af", marginTop: "4px" }}>
              {Math.round(calibrationProgress * 100)}%
            </div>
          </div>
        </div>
        {modelToggles}
      </div>
    );
  }

  // ── no data yet ───────────────────────────────────────────────────────────
  if (!hasData) {
    return (
      <div
        style={{
          backgroundColor: "#fff",
          borderRadius: "8px",
          border: "1px solid #e5e7eb",
          overflow: "hidden",
        }}
      >
        <div style={{ padding: "16px" }}>
          <SectionLabel>ML Predictions</SectionLabel>
          <div
            style={{
              color: "#9ca3af",
              fontSize: "13px",
              textAlign: "center",
              padding: "14px 0",
            }}
          >
            Start simulation to see predictions
          </div>
        </div>
        {modelToggles}
      </div>
    );
  }

  // ── live data ─────────────────────────────────────────────────────────────
  return (
    <div
      style={{
        backgroundColor: "#fff",
        borderRadius: "8px",
        border: "1px solid #e5e7eb",
        overflow: "hidden",
      }}
    >
      {/* Ensemble gauge */}
      <div style={{ padding: "16px", backgroundColor: ec!.bg }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "10px",
          }}
        >
          <span
            style={{
              fontSize: "11px",
              fontWeight: 600,
              color: "#6b7280",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            Ensemble Score
          </span>
          <span
            style={{
              fontSize: "11px",
              fontWeight: 700,
              color: ec!.text,
              padding: "2px 8px",
              borderRadius: "9999px",
              backgroundColor: "#fff",
              border: `1px solid ${ec!.bar}55`,
            }}
          >
            {ec!.label.toUpperCase()}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span
            style={{
              fontSize: "34px",
              fontWeight: 800,
              color: ec!.text,
              fontVariantNumeric: "tabular-nums",
              lineHeight: 1,
              minWidth: "80px",
            }}
          >
            {((ensemble as number) * 100).toFixed(1)}%
          </span>

          <div style={{ flex: 1 }}>
            <div
              style={{
                height: "10px",
                backgroundColor: "#e5e7eb",
                borderRadius: "5px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${(ensemble as number) * 100}%`,
                  height: "100%",
                  backgroundColor: ec!.bar,
                  borderRadius: "5px",
                  transition: "width 0.4s ease",
                }}
              />
            </div>
            {/* Zone tick marks */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: "3px",
                fontSize: "9px",
                paddingLeft: "0%",
              }}
            >
              <span style={{ color: "#22c55e" }}>0%</span>
              <span style={{ color: "#f59e0b", marginLeft: "20%" }}>20%</span>
              <span style={{ color: "#ef4444", marginLeft: "auto" }}>70%</span>
              <span style={{ color: "#9ca3af" }}>100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Per-model votes */}
      {activeModels.length > 0 && (
        <div style={{ padding: "14px 16px", borderTop: "1px solid #e5e7eb" }}>
          <SectionLabel>Model Votes</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {activeModels.map((model) => {
              const pred = preds[model.name];
              if (!pred) return null;
              const c = probStyle(pred.probability);
              const conf = pred.confidence ?? 0;
              const filledDots = Math.round(conf * 5);

              return (
                <div key={model.name}>
                  {/* Name + badge + probability */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "4px",
                    }}
                  >
                    <span style={{ fontSize: "12px", fontWeight: 500, color: "#374151" }}>
                      {model.name}
                    </span>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      {pred.classification && (
                        <span
                          style={{
                            fontSize: "10px",
                            fontWeight: 600,
                            padding: "1px 6px",
                            borderRadius: "9999px",
                            backgroundColor: c.bg,
                            color: c.text,
                          }}
                        >
                          {pred.classification}
                        </span>
                      )}
                      <span
                        style={{
                          fontSize: "13px",
                          fontWeight: 700,
                          color: c.text,
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {(pred.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Probability bar */}
                  <div
                    style={{
                      height: "6px",
                      backgroundColor: "#e5e7eb",
                      borderRadius: "3px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${pred.probability * 100}%`,
                        height: "100%",
                        backgroundColor: c.bar,
                        borderRadius: "3px",
                        transition: "width 0.3s ease",
                      }}
                    />
                  </div>

                  {/* Confidence dots + label */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginTop: "3px",
                    }}
                  >
                    <div style={{ display: "flex", gap: "2px" }}>
                      {Array.from({ length: 5 }).map((_, i) => (
                        <div
                          key={i}
                          style={{
                            width: "12px",
                            height: "3px",
                            borderRadius: "1.5px",
                            backgroundColor: i < filledDots ? "#94a3b8" : "#e5e7eb",
                          }}
                        />
                      ))}
                    </div>
                    <span style={{ fontSize: "10px", color: "#9ca3af" }}>
                      {pred.buffer_status && pred.buffer_status !== "ready"
                        ? pred.buffer_status
                        : `${(conf * 100).toFixed(0)}% conf`}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {modelToggles}
    </div>
  );
}
