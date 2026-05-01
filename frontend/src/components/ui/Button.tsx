import type { CSSProperties, ReactNode } from "react";
import {
	COLOR,
	FONT_SIZE,
	FONT_WEIGHT,
	RADIUS,
	SHADOW,
	SPACING,
} from "../../styles/tokens";

type ButtonVariant = "primary" | "danger" | "secondary" | "ghost";
type ButtonSize = "sm" | "md";

interface ButtonProps {
	children: ReactNode;
	onClick?: () => void;
	variant?: ButtonVariant;
	size?: ButtonSize;
	disabled?: boolean;
	type?: "button" | "submit";
	style?: CSSProperties;
}

const BASE_STYLE: CSSProperties = {
	border: "none",
	cursor: "pointer",
	fontFamily: "inherit",
	fontWeight: FONT_WEIGHT.MEDIUM,
	borderRadius: RADIUS.MD,
	transition: "opacity 0.15s, background-color 0.15s",
	display: "inline-flex",
	alignItems: "center",
	justifyContent: "center",
};

const VARIANT_STYLES: Record<ButtonVariant, CSSProperties> = {
	primary: {
		backgroundColor: COLOR.PRIMARY,
		color: COLOR.WHITE,
	},
	danger: {
		backgroundColor: COLOR.DANGER_BRIGHT,
		color: COLOR.WHITE,
	},
	secondary: {
		backgroundColor: COLOR.GRAY_100,
		color: COLOR.GRAY_700,
		boxShadow: SHADOW.SM,
	},
	ghost: {
		backgroundColor: "transparent",
		color: COLOR.GRAY_700,
	},
};

const SIZE_STYLES: Record<ButtonSize, CSSProperties> = {
	sm: {
		padding: `${SPACING.XS} ${SPACING.LG}`,
		fontSize: FONT_SIZE.MD,
	},
	md: {
		padding: `${SPACING.SM} ${SPACING.LG}`,
		fontSize: FONT_SIZE.BASE,
	},
};

export function Button({
	children,
	onClick,
	variant = "primary",
	size = "md",
	disabled = false,
	type = "button",
	style,
}: ButtonProps) {
	return (
		<button
			type={type}
			onClick={onClick}
			disabled={disabled}
			style={{
				...BASE_STYLE,
				...VARIANT_STYLES[variant],
				...SIZE_STYLES[size],
				cursor: disabled ? "not-allowed" : "pointer",
				opacity: disabled ? 0.6 : 1,
				...style,
			}}
		>
			{children}
		</button>
	);
}
