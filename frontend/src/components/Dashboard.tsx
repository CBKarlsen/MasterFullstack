import { useEffect, useState } from 'react';
import axios from 'axios';
import { useWebSocket } from '../hooks/useWebSocket';
import { useSimulationStore } from '../store/simulationStore';
import { TrafficLight } from './TrafficLight';
import { ModelCard } from './ModelCard';
import { ProbabilityChart } from './ProbabilityChart';
import { RawDataChart } from './RawDataChart';
import { ColumnSelector } from './ColumnSelector';
import { ModelUpload } from './ModelUpload';
import { DataFileSelector } from './DataFileSelector';
import { ControlChartGrid } from './ControlChartGrid';
import type { TrafficLight as TLType } from '../types';

export function Dashboard() {
  const {
    currentData,
    status,
    chartData,
    rawChartData,
    models,
    isCalibrating,
    calibrationProgress,
    availableColumns,
    selectedColumns,
    updateData,
    setModels,
    toggleModel,
    setSelectedColumns,
    clearData,
  } = useSimulationStore();

  // Simulation configuration
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [speed, setSpeed] = useState<number>(1);

  const { isConnected, startSimulation, stopSimulation, error } = useWebSocket(
    '/ws/simulate',
    { onMessage: updateData }
  );

  // Fetch models on mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get('/api/models');
      setModels(response.data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const handleToggleModel = (name: string, enabled: boolean) => {
    toggleModel(name, enabled);
  };

  const handleStart = () => {
    if (!selectedFile) {
      alert('Please select a data file first');
      return;
    }
    clearData();
    startSimulation(selectedFile, speed);
  };

  const handleStop = () => {
    stopSimulation();
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '24px',
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: '24px' }}>
            Pipe Clogging Detection Platform
          </h1>
          <p style={{ margin: '4px 0 0', color: '#6b7280' }}>
            Real-time FFT & ML-based anomaly detection
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {/* Speed selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ fontSize: '14px', color: '#6b7280' }}>Speed:</label>
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              disabled={isConnected}
              style={{
                padding: '6px 10px',
                borderRadius: '4px',
                border: '1px solid #e5e7eb',
                fontSize: '14px',
              }}
            >
              <option value={0.5}>0.5x</option>
              <option value={1}>1x (Real-time)</option>
              <option value={2}>2x</option>
              <option value={5}>5x</option>
              <option value={10}>10x</option>
              <option value={20}>20x</option>
              <option value={50}>50x</option>
            </select>
          </div>

          <span
            style={{
              padding: '4px 12px',
              borderRadius: '9999px',
              fontSize: '14px',
              backgroundColor: isConnected ? '#dcfce7' : '#fee2e2',
              color: isConnected ? '#166534' : '#991b1b',
            }}
          >
            {isConnected ? 'Running' : 'Stopped'}
          </span>
          <button
            onClick={isConnected ? handleStop : handleStart}
            disabled={!selectedFile && !isConnected}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              backgroundColor: isConnected ? '#ef4444' : '#3b82f6',
              color: '#fff',
              fontWeight: 500,
              cursor: !selectedFile && !isConnected ? 'not-allowed' : 'pointer',
              opacity: !selectedFile && !isConnected ? 0.6 : 1,
            }}
          >
            {isConnected ? 'Stop' : 'Start Simulation'}
          </button>
        </div>
      </div>

      {error && (
        <div
          style={{
            padding: '12px',
            marginBottom: '16px',
            backgroundColor: '#fee2e2',
            color: '#991b1b',
            borderRadius: '6px',
          }}
        >
          {error}
        </div>
      )}

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '24px' }}>
        {/* Left Column - Charts and Status */}
        <div>
          {/* Status Card */}
          <div
            style={{
              padding: '20px',
              backgroundColor: '#fff',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              marginBottom: '20px',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <TrafficLight
                color={(currentData?.traffic_light || 'gray') as TLType}
                message={currentData?.light_msg || status}
              />
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '32px', fontWeight: 'bold' }}>
                  {currentData?.ensemble_probability !== undefined
                    ? `${(currentData.ensemble_probability * 100).toFixed(1)}%`
                    : '—'}
                </div>
                <div style={{ fontSize: '14px', color: '#6b7280' }}>
                  Ensemble Probability
                </div>
              </div>
            </div>

            {/* Current Values */}
            {currentData && currentData.time !== undefined && (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '16px',
                  marginTop: '20px',
                  paddingTop: '16px',
                  borderTop: '1px solid #e5e7eb',
                }}
              >
                <MetricBox
                  label="Time"
                  value={`${currentData.time?.toFixed(1)}s`}
                />
                <MetricBox
                  label="Spectral Slope"
                  value={currentData.spectral_slope?.toFixed(2) || '—'}
                />
                <MetricBox
                  label="Static Score"
                  value={currentData.static_score?.toFixed(4) || '—'}
                />
                <MetricBox
                  label="Composite"
                  value={currentData.composite_score?.toFixed(4) || '—'}
                />
              </div>
            )}
          </div>

          {/* Probability Chart */}
          <div
            style={{
              padding: '20px',
              backgroundColor: '#fff',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
            }}
          >
            <h2 style={{ margin: '0 0 16px', fontSize: '18px' }}>
              Model Probabilities Over Time
            </h2>
            <ProbabilityChart
              data={chartData}
              models={models}
              isCalibrating={isCalibrating}
              calibrationProgress={calibrationProgress}
            />
          </div>

          {/* Raw Data Chart */}
          <div
            style={{
              padding: '20px',
              backgroundColor: '#fff',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              marginTop: '20px',
            }}
          >
            <h2 style={{ margin: '0 0 16px', fontSize: '18px' }}>
              Physics Data
            </h2>
            <RawDataChart
              data={rawChartData}
              selectedColumns={selectedColumns}
              height={250}
            />
          </div>

          {/* Control Charts Grid */}
          <div style={{ marginTop: '20px' }}>
            <ControlChartGrid
              rawData={rawChartData}
              selectedColumns={selectedColumns}
              chartData={chartData}
              maxDataPoints={200000}
              chartHeight={180}
            />
          </div>
        </div>

        {/* Right Column - Data & Models */}
        <div>
          {/* Data File Selector */}
          <div style={{ marginBottom: '20px' }}>
            <DataFileSelector
              selectedFile={selectedFile}
              onSelectFile={setSelectedFile}
              disabled={isConnected}
            />
          </div>

          {/* Column Selector */}
          <div style={{ marginBottom: '20px' }}>
            <ColumnSelector
              availableColumns={availableColumns}
              selectedColumns={selectedColumns}
              onSelectionChange={setSelectedColumns}
              disabled={false}
            />
          </div>

          {/* Model Upload */}
          <div
            style={{
              padding: '20px',
              backgroundColor: '#fff',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              marginBottom: '20px',
            }}
          >
            <ModelUpload onUploadComplete={fetchModels} />
          </div>

          {/* Model List */}
          <div
            style={{
              padding: '20px',
              backgroundColor: '#fff',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
              }}
            >
              <h2 style={{ margin: 0, fontSize: '18px' }}>
                Models ({models.length})
              </h2>
              <button
                onClick={fetchModels}
                style={{
                  padding: '4px 8px',
                  fontSize: '12px',
                  border: '1px solid #e5e7eb',
                  borderRadius: '4px',
                  backgroundColor: '#fff',
                  cursor: 'pointer',
                }}
              >
                Refresh
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {models.length === 0 ? (
                <p style={{ color: '#6b7280', textAlign: 'center', padding: '20px' }}>
                  No models loaded. Start the backend server first.
                </p>
              ) : (
                models.map((model) => (
                  <ModelCard
                    key={model.name}
                    model={model}
                    prediction={currentData?.models?.[model.name]}
                    onToggle={handleToggleModel}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: '12px', color: '#6b7280' }}>{label}</div>
      <div style={{ fontSize: '18px', fontWeight: 600 }}>{value}</div>
    </div>
  );
}
