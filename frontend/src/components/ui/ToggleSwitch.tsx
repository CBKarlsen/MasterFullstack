import { COLOR, RADIUS, SHADOW } from "../../styles/tokens";

interface ToggleSwitchProps {
	enabled: boolean;
	loading: boolean;
	onClick: () => void;
}

export function ToggleSwitch({ enabled, loading, onClick }: ToggleSwitchProps) {
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
				backgroundColor: enabled ? COLOR.SUCCESS : COLOR.GRAY_300,
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
					borderRadius: RADIUS.CIRCLE,
					backgroundColor: COLOR.WHITE,
					transition: "left 0.2s",
					boxShadow: SHADOW.SM,
				}}
			/>
		</button>
	);
}
