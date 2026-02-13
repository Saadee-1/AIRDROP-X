"""
AIRDROP-X Tactical UI Theme
Military-grade unified color system and typography scale.

This module provides a single source of truth for all UI styling constants,
ensuring visual consistency and professional military-grade appearance across
all tabs and components.
"""

# =============================================================================
# COLOR PALETTE - Tactical Dark Theme
# =============================================================================

# --- BACKGROUNDS ---
BG_MAIN = "#0a0c0a"        # Main canvas background (darkest)
BG_PANEL = "#0f120f"       # Panel/container backgrounds
BG_INPUT = "#1c201c"       # Input fields, text boxes

# --- TEXT HIERARCHY ---
TEXT_PRIMARY = "#e0ece0"   # Primary text (high contrast)
TEXT_SECONDARY = "#c0d0c0"  # Secondary text (medium contrast)
TEXT_LABEL = "#6b8e6b"     # Labels, hints, muted text

# --- ACCENT COLORS ---
ACCENT_GO = "#00ff41"      # Green - GO/DROP/success indicator
ACCENT_NO_GO = "#ff3333"   # Red - NO-GO/abort/error indicator
ACCENT_WARN = "#e6b800"    # Yellow/orange - warnings, cautions
ACCENT_INFO = "#4a7c4a"    # Muted green - informational highlights

# --- BORDERS & DIVIDERS ---
BORDER_SUBTLE = "#2a3a2a"  # Standard panel borders
BORDER_FOCUS = "#3a4a3a"   # Focused/active element borders

# --- CHART & VISUALIZATION ---
SCATTER_PRIMARY = "#00ff41"  # Primary scatter plot color
CEP_CIRCLE = "#4a7c4a"       # CEP circle overlay
MEAN_MARKER = "#ffffff"      # Mean position marker

# =============================================================================
# TYPOGRAPHY SCALE
# =============================================================================

FONT_FAMILY = "monospace"  # Military HUD aesthetic

# Standard hierarchy (points)
FONT_SIZE_BANNER = 28      # Decision banners (special case)
FONT_SIZE_H1 = 14          # Section headers
FONT_SIZE_H2 = 12          # Subsection headers
FONT_SIZE_H3 = 10          # Panel titles
FONT_SIZE_BODY = 9         # Body text, primary labels
FONT_SIZE_CAPTION = 8      # Captions, hints, fine print
FONT_SIZE_SMALL = 7        # Extremely small annotations

# =============================================================================
# SPACING GRID (in normalized axes coordinates, 0-1)
# =============================================================================
# Based on 8px grid at 800px window width → 0.01 ≈ 8px

SPACING_TINY = 0.01        # 8px - minimal gap
SPACING_SMALL = 0.02       # 16px - standard gap
SPACING_MEDIUM = 0.04      # 32px - section spacing
SPACING_LARGE = 0.06       # 48px - major divisions

# =============================================================================
# COMPONENT CONSTANTS
# =============================================================================

# Standard panel border width
BORDER_WIDTH = 1

# Button hover color (10% lighter than BG_INPUT)
BUTTON_HOVER = "#252925"

# Input field height (normalized)
INPUT_HEIGHT = 0.06

# Tab button dimensions
TAB_WIDTH = 0.14
TAB_HEIGHT = 0.06
TAB_GAP = 0.02
