import { useState, useCallback } from 'react';
import axios from 'axios';

interface ModelUploadProps {
  onUploadComplete: () => void;
}

export function ModelUpload({ onUploadComplete }: ModelUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form fields
  const [name, setName] = useState('');
  const [author, setAuthor] = useState('');
  const [description, setDescription] = useState('');

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        await uploadFile(file);
      }
    },
    [name, author, description]
  );

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file: File) => {
    // Validate extension
    const validExtensions = ['.pkl', '.pt', '.pth', '.h5'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!validExtensions.includes(ext)) {
      setError(`Invalid file type. Allowed: ${validExtensions.join(', ')}`);
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    if (name) formData.append('name', name);
    if (author) formData.append('author', author);
    if (description) formData.append('description', description);

    try {
      await axios.post('/api/models/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Reset form
      setName('');
      setAuthor('');
      setDescription('');

      onUploadComplete();
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || 'Upload failed');
      } else {
        setError('Upload failed');
      }
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ marginBottom: '24px' }}>
      <h3 style={{ margin: '0 0 12px', fontSize: '16px' }}>Upload New Model</h3>

      {/* Optional metadata fields */}
      <div style={{ display: 'grid', gap: '8px', marginBottom: '12px' }}>
        <input
          type="text"
          placeholder="Model name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{
            padding: '8px 12px',
            border: '1px solid #e5e7eb',
            borderRadius: '4px',
            fontSize: '14px',
          }}
        />
        <input
          type="text"
          placeholder="Author (optional)"
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          style={{
            padding: '8px 12px',
            border: '1px solid #e5e7eb',
            borderRadius: '4px',
            fontSize: '14px',
          }}
        />
        <input
          type="text"
          placeholder="Description (optional)"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{
            padding: '8px 12px',
            border: '1px solid #e5e7eb',
            borderRadius: '4px',
            fontSize: '14px',
          }}
        />
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${isDragging ? '#3b82f6' : '#e5e7eb'}`,
          borderRadius: '8px',
          padding: '24px',
          textAlign: 'center',
          backgroundColor: isDragging ? '#eff6ff' : '#f9fafb',
          transition: 'all 0.2s',
          cursor: 'pointer',
        }}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".pkl,.pt,.pth,.h5"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        {uploading ? (
          <p style={{ margin: 0, color: '#3b82f6' }}>Uploading...</p>
        ) : (
          <>
            <p style={{ margin: 0, fontWeight: 500 }}>
              Drop model file here or click to browse
            </p>
            <p style={{ margin: '8px 0 0', fontSize: '13px', color: '#6b7280' }}>
              Supports .pkl (sklearn), .pt/.pth (PyTorch), .h5 (TensorFlow)
            </p>
          </>
        )}
      </div>

      {error && (
        <p style={{ margin: '8px 0 0', color: '#ef4444', fontSize: '14px' }}>
          {error}
        </p>
      )}
    </div>
  );
}
