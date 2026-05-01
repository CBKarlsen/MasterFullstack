import type { ReactNode } from "react";
import { useState } from "react";
import { COLOR, FONT_SIZE, RADIUS, SHADOW } from "../../styles/tokens";

interface TooltipProps {
	text: string;
	children?: ReactNode;
}

export function Tooltip({ text, children }: TooltipProps) {
	const [visible, setVisible] = useState(false);

	return (
		<span
			style={{
				position: "relative",
				display: "inline-flex",
				alignItems: "center",
				gap: "4px",
			}}
			onMouseEnter={() => setVisible(true)}
			onMouseLeave={() => setVisible(false)}
		>
			{children}
			<span
				aria-label={text}
				style={{
					display: "inline-flex",
					alignItems: "center",
					justifyContent: "center",
					width: "14px",
					height: "14px",
					borderRadius: RADIUS.CIRCLE,
					backgroundColor: COLOR.GRAY_300,
					color: COLOR.GRAY_700,
					fontSize: "10px",
					fontWeight: 600,
					cursor: "default",
					flexShrink: 0,
					lineHeight: 1,
				}}
			>
				?
			</span>
			{visible && (
				<span
					role="tooltip"
					style={{
						position: "absolute",
						bottom: "calc(100% + 6px)",
						left: "50%",
						transform: "translateX(-50%)",
						backgroundColor: COLOR.GRAY_800,
						color: COLOR.WHITE,
						fontSize: FONT_SIZE.XS,
						lineHeight: 1.5,
						padding: "6px 10px",
						borderRadius: RADIUS.MD,
						boxShadow: SHADOW.MD,
						whiteSpace: "normal",
						width: "220px",
						zIndex: 100,
						pointerEvents: "none",
					}}
				>
					{text}
				</span>
			)}
		</span>
	);
}
