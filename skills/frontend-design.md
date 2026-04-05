# Frontend Design Skill

Before writing any UI code, commit to a BOLD aesthetic direction:
- Purpose: What problem does this interface solve? Who uses it?
- Tone: Pick one extreme and execute it with full intentionality.
  Options: brutally minimal, maximalist, retro-futuristic, luxury/refined,
  industrial/utilitarian, editorial, brutalist, art deco/geometric, etc.
- Differentiation: What makes this interface UNFORGETTABLE?

## Aesthetics Rules

### Typography
- NEVER use Inter, Roboto, Arial, or system fonts
- Load a distinctive Google Font via <link> or @import
- Pair a strong display font for headers with a clean monospace for data/numbers

### Color & Theme
- Use CSS variables for full consistency
- Dominant color + sharp accent outperforms evenly distributed palettes
- Commit fully — no timid half-choices

### Motion
- CSS animations for page load (staggered reveals with animation-delay)
- Hover states that surprise
- For Streamlit: use CSS transitions via st.markdown injection

### Spatial Composition
- Asymmetry, overlap, grid-breaking elements
- Generous negative space OR controlled density — pick one

### Backgrounds & Depth
- Gradient meshes, noise textures, geometric patterns, layered transparencies
- Dramatic shadows, decorative borders
- NEVER flat solid color backgrounds on important surfaces

## For Streamlit Specifically
- Inject global CSS via st.markdown("""<style>...</style>""", unsafe_allow_html=True)
- Override: .stApp, .stTabs, .stMetric, .stSidebar, dataframe cells
- Load Google Fonts in the same CSS injection block with @import url(...)
- Plotly charts: always set template="plotly_dark" + custom color_discrete_sequence
- Use st.columns() creatively — unequal widths, nested columns for card layouts
- Styled cards: st.markdown(<div class="card">...</div>, unsafe_allow_html=True)

## What to NEVER Do
- Purple gradients on white — cliché AI aesthetic
- Default Streamlit grey/white theme with no overrides
- Default blue Plotly charts
- Cookie-cutter metric boxes with no styling
- Same font choices as every other AI-generated dashboard
