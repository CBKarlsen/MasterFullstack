import axios from "axios";
import { useCallback, useEffect, useState } from "react";
import type { DataFile } from "../types";

interface DataFileSelectorProps {
	selectedFile: string | null;
	onSelectFile: (filename: string) => void;
	disabled?: boolean;
}

export function DataFileSelector({
	selectedFile,
	onSelectFile,
	disabled = false,
}: DataFileSelectorProps) {
	const [files, setFiles] = useState<DataFile[]>([]);
	const [loading, setLoading] = useState(true);
	const [uploading, setUploading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const fetchFiles = useCallback(async () => {
		try {
			setLoading(true);
			const response = await axios.get<DataFile[]>("/api/data");
			setFiles(response.data);
			setError(null);

			// Auto-select first file if none selected
			if (!selectedFile && response.data.length > 0) {
				onSelectFile(response.data[0].name);
			}
		} catch (err) {
			setError("Failed to load data files");
			console.error(err);
		} finally {
			setLoading(false);
		}
	}, [selectedFile, onSelectFile]);

	useEffect(() => {
		fetchFiles();
	}, [fetchFiles]);

	const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (!file) return;

		// Validate extension
		const ext = file.name.split(".").pop()?.toLowerCase();
		if (!["csv", "xlsx"].includes(ext || "")) {
			setError("Only CSV and Excel files are allowed");
			return;
		}

		setUploading(true);
		setError(null);

		const formData = new FormData();
		formData.append("file", file);

		try {
			await axios.post("/api/data/upload", formData, {
				headers: { "Content-Type": "multipart/form-data" },
			});

			// Refresh file list and select the new file
			await fetchFiles();
			onSelectFile(file.name);
		} catch (err) {
			setError("Failed to upload file");
			console.error(err);
		} finally {
			setUploading(false);
			// Reset input
			e.target.value = "";
		}
	};

	return (
		<div
			style={{
				padding: "16px",
				backgroundColor: "#fff",
				borderRadius: "8px",
				border: "1px solid #e5e7eb",
			}}
		>
			<div
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					marginBottom: "12px",
				}}
			>
				<h3 style={{ margin: 0, fontSize: "16px" }}>Data Source</h3>
				<label
					style={{
						padding: "6px 12px",
						fontSize: "13px",
						backgroundColor: "#3b82f6",
						color: "#fff",
						borderRadius: "4px",
						cursor: uploading || disabled ? "not-allowed" : "pointer",
						opacity: uploading || disabled ? 0.6 : 1,
					}}
				>
					{uploading ? "Uploading..." : "Upload File"}
					<input
						type="file"
						accept=".csv,.xlsx"
						onChange={handleFileUpload}
						disabled={uploading || disabled}
						style={{ display: "none" }}
					/>
				</label>
			</div>

			{error && (
				<div
					style={{
						padding: "8px",
						marginBottom: "12px",
						backgroundColor: "#fee2e2",
						color: "#991b1b",
						borderRadius: "4px",
						fontSize: "13px",
					}}
				>
					{error}
				</div>
			)}

			{loading ? (
				<div style={{ padding: "20px", textAlign: "center", color: "#6b7280" }}>
					Loading files...
				</div>
			) : files.length === 0 ? (
				<div style={{ padding: "20px", textAlign: "center", color: "#6b7280" }}>
					No data files found. Upload a CSV or Excel file to get started.
				</div>
			) : (
				<div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
					{files.map((file) => (
						<button
							key={file.name}
							onClick={() => onSelectFile(file.name)}
							disabled={disabled}
							style={{
								display: "flex",
								justifyContent: "space-between",
								alignItems: "center",
								padding: "10px 12px",
								border:
									selectedFile === file.name
										? "2px solid #3b82f6"
										: "1px solid #e5e7eb",
								borderRadius: "6px",
								backgroundColor:
									selectedFile === file.name ? "#eff6ff" : "#fff",
								cursor: disabled ? "not-allowed" : "pointer",
								textAlign: "left",
								opacity: disabled ? 0.6 : 1,
							}}
						>
							<div
								style={{ display: "flex", alignItems: "center", gap: "8px" }}
							>
								{/* Icon for folder vs file */}
								<span style={{ fontSize: "16px" }}>
									{file.is_folder ? "📁" : "📄"}
								</span>
								<div>
									<div
										style={{
											fontWeight: selectedFile === file.name ? 600 : 400,
											fontSize: "14px",
										}}
									>
										{file.name}
									</div>
									<div style={{ fontSize: "12px", color: "#6b7280" }}>
										{file.is_folder
											? `${file.file_count} files • ${file.size_human}`
											: file.size_human}
									</div>
								</div>
							</div>
							{selectedFile === file.name && (
								<div
									style={{
										width: "8px",
										height: "8px",
										borderRadius: "50%",
										backgroundColor: "#3b82f6",
									}}
								/>
							)}
						</button>
					))}
				</div>
			)}

			<button
				onClick={fetchFiles}
				disabled={loading}
				style={{
					marginTop: "12px",
					padding: "6px 12px",
					fontSize: "12px",
					border: "1px solid #e5e7eb",
					borderRadius: "4px",
					backgroundColor: "#fff",
					cursor: "pointer",
					width: "100%",
				}}
			>
				Refresh List
			</button>
		</div>
	);
}
