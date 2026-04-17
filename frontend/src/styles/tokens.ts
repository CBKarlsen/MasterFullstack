export const COLOR = {
  // Brand
  PRIMARY: '#3b82f6',
  PRIMARY_HOVER: '#2563eb',

  // Semantic
  SUCCESS: '#16a34a',
  SUCCESS_BG: '#dcfce7',
  SUCCESS_TEXT: '#166534',
  SUCCESS_BRIGHT: '#22c55e',
  SUCCESS_SUBTLE: '#f0fdf4',

  WARNING: '#f59e0b',
  WARNING_BG: '#fffbeb',
  WARNING_TEXT: '#d97706',
  WARNING_ALT: '#eab308',

  DANGER: '#dc2626',
  DANGER_BRIGHT: '#ef4444',
  DANGER_BG: '#fef2f2',
  DANGER_BORDER: '#fca5a5',
  DANGER_TEXT: '#991b1b',

  // Neutrals
  WHITE: '#ffffff',
  GRAY_50: '#f9fafb',
  GRAY_100: '#f3f4f6',
  GRAY_200: '#e5e7eb',
  GRAY_300: '#d1d5db',
  GRAY_400: '#9ca3af',
  GRAY_500: '#6b7280',
  GRAY_700: '#374151',
  GRAY_800: '#1f2937',
  GRAY_900: '#111827',

  // Accents
  INFO: '#0284c7',
  PURPLE: '#7c3aed',
} as const

export const RADIUS = {
  SM: '4px',
  MD: '6px',
  LG: '8px',
  XL: '10px',
  PILL: '9999px',
  CIRCLE: '50%',
} as const

export const FONT_SIZE = {
  XS: '11px',
  SM: '12px',
  MD: '13px',
  BASE: '14px',
  LG: '16px',
  XL: '18px',
  XXL: '24px',
  DISPLAY: '32px',
} as const

export const FONT_WEIGHT = {
  NORMAL: 400,
  MEDIUM: 500,
  SEMIBOLD: 600,
  BOLD: 700,
} as const

export const SPACING = {
  XS: '4px',
  SM: '8px',
  MD: '12px',
  LG: '16px',
  XL: '20px',
  XXL: '24px',
  XXXL: '32px',
} as const

export const SHADOW = {
  SM: '0 1px 2px rgba(0,0,0,0.1)',
  MD: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06)',
  SUBTLE: '0 2px 4px rgba(0,0,0,0.05)',
} as const

export const BORDER = {
  DEFAULT: `1px solid ${COLOR.GRAY_200}`,
} as const
