import type { ReactNode } from 'react'
import { COLOR, FONT_SIZE, FONT_WEIGHT } from '../../styles/tokens'

interface SectionLabelProps {
  children: ReactNode
}

export function SectionLabel({ children }: SectionLabelProps) {
  return (
    <div
      style={{
        fontSize: FONT_SIZE.XS,
        fontWeight: FONT_WEIGHT.SEMIBOLD,
        color: COLOR.GRAY_400,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        marginBottom: '10px',
      }}
    >
      {children}
    </div>
  )
}
