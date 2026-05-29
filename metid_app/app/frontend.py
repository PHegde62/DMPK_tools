"""
app/frontend.py
================================================================================
MetID — Premium Biotech Dashboard
================================================================================
Executive-grade dark/slate interface for Metabolite Identification and
Soft Spot Analysis.  Communicates with the FastAPI backend over HTTP.

Features
--------
  • streamlit-ketcher molecule sketcher with SMILES text fallback
  • Three-panel layout: Analytics | Soft Spot Map | Metabolic Tree
  • Animated metric cards, glowing atom highlights, confidence-tier badges
  • Full error handling for timeouts, invalid SMILES, and backend failures

Running
-------
  streamlit run app/frontend.py
  METID_API_URL=https://api.myco.io streamlit run app/frontend.py
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
import streamlit as st

# ── Runtime config ────────────────────────────────────────────────────────────
API_BASE = os.getenv("METID_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT  = float(os.getenv("REQUEST_TIMEOUT", "45"))

# ── Example compounds ─────────────────────────────────────────────────────────
EXAMPLES = {
    "Aspirin":     ("CC(=O)Oc1ccccc1C(=O)O",   "Analgesic / COX inhibitor"),
    "Ibuprofen":   ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "NSAID — benzylic CYP2C9 site"),
    "Paracetamol": ("CC(=O)Nc1ccc(O)cc1",        "Phenolic OH — glucuronidation"),
    "Caffeine":    ("Cn1cnc2c1c(=O)n(c(=O)n2C)C","N-methyl CYP1A2 substrate"),
    "Lidocaine":   ("CCN(CC)CC(=O)Nc1c(C)cccc1C","Amide local anaesthetic"),
    "Omeprazole":  ("COc1ccc2[nH]c(nc2c1)Cc1nc2cc(OC)ccc2n1","PPI — sulfoxide SOM"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MetID · Metabolite Intelligence",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system CSS ─────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
  --bg-base:      #0a0f1e;
  --bg-card:      #0f172a;
  --bg-raised:    #1e293b;
  --bg-hover:     #263348;
  --border:       #1e3a5f;
  --border-glow:  #38bdf8;
  --sky-400:      #38bdf8;
  --sky-300:      #7dd3fc;
  --emerald-400:  #34d399;
  --amber-400:    #fbbf24;
  --rose-400:     #fb7185;
  --violet-400:   #a78bfa;
  --slate-300:    #cbd5e1;
  --slate-400:    #94a3b8;
  --slate-500:    #64748b;
  --white:        #f1f5f9;
  --font-display: 'Syne', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
  --radius:       10px;
  --radius-lg:    16px;
  --shadow:       0 4px 24px rgba(0,0,0,0.45);
  --glow-sky:     0 0 20px rgba(56,189,248,0.25);
  --glow-rose:    0 0 16px rgba(251,113,133,0.30);
}

/* ── Reset ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--font-mono) !important;
  background-color: var(--bg-base) !important;
  color: var(--slate-300) !important;
}
.stApp { background: var(--bg-base) !important; }
.stApp header { display: none !important; }
.block-container { padding: 0 1.5rem 3rem !important; max-width: 100% !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--slate-300) !important; }

/* ── Hero banner ────────────────────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #060d1f 0%, #0c1a35 40%, #091628 100%);
  border-bottom: 1px solid var(--border);
  padding: 2rem 2.5rem 1.8rem;
  margin: 0 -1.5rem 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: -40%; right: -5%;
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 65%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute; bottom: -50%; left: 15%;
  width: 350px; height: 350px;
  background: radial-gradient(circle, rgba(52,211,153,0.05) 0%, transparent 60%);
  pointer-events: none;
}
.hero-title {
  font-family: var(--font-display) !important;
  font-size: 2.6rem; font-weight: 800;
  background: linear-gradient(90deg, var(--sky-300) 0%, var(--emerald-400) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.3rem;
  line-height: 1.1;
  letter-spacing: -1px;
}
.hero-sub {
  font-family: var(--font-mono) !important;
  font-size: 0.75rem; color: var(--slate-500) !important;
  text-transform: uppercase; letter-spacing: 0.14em;
}
.hero-badges { margin-top: 0.8rem; display: flex; gap: 8px; flex-wrap: wrap; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(56,189,248,0.08);
  border: 1px solid rgba(56,189,248,0.25);
  color: var(--sky-300) !important;
  font-family: var(--font-mono); font-size: 0.68rem;
  padding: 3px 10px; border-radius: 20px; letter-spacing: 0.06em;
}

/* ── Section labels ─────────────────────────────────────────────────────────── */
.sec-label {
  font-family: var(--font-mono) !important;
  font-size: 0.65rem !important; font-weight: 500 !important;
  letter-spacing: 0.16em !important; text-transform: uppercase !important;
  color: var(--sky-400) !important;
  border-bottom: 1px solid rgba(56,189,248,0.18) !important;
  padding-bottom: 0.5rem !important; margin-bottom: 0.8rem !important;
}

/* ── Cards ──────────────────────────────────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.2rem 1.4rem;
  box-shadow: var(--shadow);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
  border-color: rgba(56,189,248,0.35);
  box-shadow: var(--shadow), var(--glow-sky);
}

/* ── Metric cards ───────────────────────────────────────────────────────────── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 1.4rem; }
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  text-align: center;
  transition: all 0.2s;
}
.metric-card:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 14px rgba(56,189,248,0.15);
  transform: translateY(-2px);
}
.metric-val {
  font-family: var(--font-display) !important;
  font-size: 2.2rem; font-weight: 700; line-height: 1;
}
.metric-lbl {
  font-size: 0.66rem; color: var(--slate-500) !important;
  text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;
}
.metric-sky   { color: var(--sky-400); }
.metric-em    { color: var(--emerald-400); }
.metric-am    { color: var(--amber-400); }
.metric-rose  { color: var(--rose-400); }

/* ── Molecule SVG panel ─────────────────────────────────────────────────────── */
.mol-frame {
  background: #ffffff;
  border-radius: var(--radius);
  padding: 10px;
  display: inline-block;
  box-shadow: 0 8px 32px rgba(0,0,0,0.55), var(--glow-sky);
  transition: box-shadow 0.3s;
}
.mol-frame:hover { box-shadow: 0 8px 40px rgba(0,0,0,0.65), 0 0 28px rgba(56,189,248,0.30); }

/* ── Soft spot list ─────────────────────────────────────────────────────────── */
.spot-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(30,58,95,0.6);
}
.spot-row:last-child { border-bottom: none; }
.spot-rank {
  width: 24px; height: 24px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 700;
  background: rgba(251,113,133,0.12); border: 1px solid rgba(251,113,133,0.3);
  color: var(--rose-400) !important;
}
.spot-body { flex: 1; min-width: 0; }
.spot-atom { font-size: 0.85rem; font-weight: 600; color: var(--white) !important; }
.spot-rule { font-size: 0.7rem; color: var(--slate-500) !important; margin-top: 1px; }
.spot-vi-wrap { text-align: right; min-width: 70px; }
.spot-vi-num  { font-size: 0.82rem; font-weight: 700; color: var(--rose-400) !important; }
.spot-bar-track { background: rgba(30,58,95,0.8); border-radius: 3px; height: 5px; margin-top: 3px; overflow: hidden; }
.spot-bar-fill  { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--rose-400), var(--amber-400)); }

/* ── Confidence tier badges ─────────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--font-mono); font-size: 0.65rem; font-weight: 600;
  padding: 2px 9px; border-radius: 12px; letter-spacing: 0.04em;
  white-space: nowrap;
}
.badge-consensus {
  background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3);
  color: var(--emerald-400) !important;
}
.badge-rule {
  background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.25);
  color: var(--sky-400) !important;
}
.badge-dl {
  background: rgba(167,139,250,0.10); border: 1px solid rgba(167,139,250,0.25);
  color: var(--violet-400) !important;
}
.badge-p1 {
  background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.20);
  color: var(--sky-300) !important;
}
.badge-p2 {
  background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.25);
  color: var(--amber-400) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--sky-300) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.9rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--sky-400) !important;
  box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────────── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--border-glow) !important;
  color: var(--sky-300) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-radius: var(--radius) !important;
  transition: all 0.18s !important;
}
.stButton > button:hover {
  background: rgba(56,189,248,0.10) !important;
  box-shadow: var(--glow-sky) !important;
  transform: translateY(-1px);
}

/* ── Dataframe ──────────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────────── */
details summary {
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  color: var(--sky-400) !important;
  cursor: pointer;
}

/* ── Spinner ────────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--sky-400) !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--sky-400); }

/* ── Property pill grid ─────────────────────────────────────────────────────── */
.prop-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 0.8rem; }
.prop-item {
  background: var(--bg-raised); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px;
}
.prop-key  { font-size: 0.62rem; color: var(--slate-500) !important; text-transform: uppercase; letter-spacing: 0.07em; }
.prop-val  { font-size: 0.88rem; font-weight: 600; color: var(--white) !important; margin-top: 1px; }

/* ── Error / warning banners ────────────────────────────────────────────────── */
.err-banner {
  background: rgba(251,113,133,0.08);
  border-left: 3px solid var(--rose-400);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 0.9rem 1.2rem;
  margin: 0.8rem 0;
  font-size: 0.84rem;
  color: var(--rose-400) !important;
}
.warn-banner {
  background: rgba(251,191,36,0.06);
  border-left: 3px solid var(--amber-400);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: 0.8rem 1.1rem;
  font-size: 0.82rem;
  color: var(--amber-400) !important;
  margin: 0.5rem 0;
}

/* ── Empty state ────────────────────────────────────────────────────────────── */
.empty-state {
  text-align: center; padding: 3.5rem 1rem;
  color: var(--slate-500) !important;
}
.empty-icon { font-size: 3rem; opacity: 0.25; margin-bottom: 0.8rem; }

/* ── Footer ─────────────────────────────────────────────────────────────────── */
.footer {
  text-align: center; font-size: 0.68rem;
  color: var(--slate-500) !important;
  border-top: 1px solid var(--border);
  padding: 1.5rem 0 0.5rem;
  margin-top: 2.5rem;
  letter-spacing: 0.06em;
}

/* ── Selectbox ──────────────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-raised) !important;
  border-color: var(--border) !important;
  color: var(--slate-300) !important;
}
</style>
"""


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "smiles":        "",
        "result":        None,
        "svg":           None,
        "error":         None,
        "backend_ok":    None,
        "met_filter":    "all",
        "show_all_mets": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API helpers ───────────────────────────────────────────────────────────────
def _check_backend() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _post_predict(smiles: str, p1: int, p2: int, max_m: int, top_n: int):
    try:
        r = httpx.post(
            f"{API_BASE}/predict",
            json=dict(smiles=smiles, phase1_cycles=p1, phase2_cycles=p2,
                      max_metabolites=max_m, top_soft_spots=top_n, include_svg=False),
            timeout=TIMEOUT,
        )
    except httpx.ConnectError:
        return None, f"**Backend unreachable** at `{API_BASE}`.  Run `uvicorn app.main:app --port 8000`."
    except httpx.TimeoutException:
        return None, f"**Request timed out** after {TIMEOUT:.0f} s.  Try fewer cycles or increase `REQUEST_TIMEOUT`."
    except Exception as e:
        return None, f"**Network error:** `{e}`"

    if r.status_code == 422:
        try:
            detail = r.json().get("detail", [])
            msg = "; ".join(d.get("msg", str(d)) for d in detail) if isinstance(detail, list) else str(detail)
        except Exception:
            msg = r.text[:300]
        return None, f"**Invalid input (422):** {msg}"
    if not r.is_success:
        try:
            msg = r.json().get("detail", r.text[:300])
        except Exception:
            msg = r.text[:300]
        return None, f"**API error {r.status_code}:** {msg}"
    try:
        return r.json(), None
    except Exception:
        return None, "**Malformed JSON response from backend.**"


def _post_render(smiles: str, indices: List[int], w: int, h: int,
                 color: List[float], scores: bool):
    try:
        r = httpx.post(
            f"{API_BASE}/render-soft-spots",
            json=dict(smiles=smiles, highlight_indices=indices,
                      width=w, height=h, highlight_color=color,
                      show_scores=scores, show_atom_indices=False),
            timeout=TIMEOUT,
        )
        return (r.text, None) if r.is_success else (None, f"Render error {r.status_code}")
    except Exception as e:
        return None, f"Render network error: {e}"


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _badge_tier(tier: str) -> str:
    t = tier.lower()
    if "consensus" in t:
        return '<span class="badge badge-consensus">✦ Consensus</span>'
    if "rule" in t:
        return '<span class="badge badge-rule">◈ Rule-Only</span>'
    if "dl" in t:
        return '<span class="badge badge-dl">◉ DL-Only</span>'
    return f'<span class="badge badge-rule">{tier[:18]}</span>'


def _badge_phase(phase: int) -> str:
    if phase == 1:
        return '<span class="badge badge-p1">Phase I</span>'
    if phase == 2:
        return '<span class="badge badge-p2">Phase II</span>'
    return '<span class="badge badge-dl">Phase ?</span>'


def _metric(val: Any, label: str, cls: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-val {cls}">{val}</div>'
        f'<div class="metric-lbl">{label}</div>'
        f'</div>'
    )


def _spot_rows(spots: List[Dict]) -> str:
    if not spots:
        return "<p style='color:var(--slate-500);font-size:0.82rem;'>No soft spots identified.</p>"
    rows = []
    for i, s in enumerate(spots, 1):
        vi  = s.get("vulnerability_index", s.get("score", 0) * 100)
        pct = min(100, max(0, int(vi)))
        rows.append(f"""
        <div class="spot-row">
          <div class="spot-rank">{i}</div>
          <div class="spot-body">
            <div class="spot-atom">{s['atom_symbol']} <span style="color:var(--slate-500)">atom&nbsp;{s['atom_index']}</span></div>
            <div class="spot-rule">{s.get('rule_name','').replace('_',' ')}</div>
          </div>
          <div class="spot-vi-wrap">
            <div class="spot-vi-num">{vi:.1f}%</div>
            <div class="spot-bar-track"><div class="spot-bar-fill" style="width:{pct}%"></div></div>
          </div>
        </div>""")
    return "".join(rows)


def _prop_grid(p: Dict) -> str:
    items = [
        ("Formula",    p.get("molecular_formula", "—")),
        ("MW",         f'{p.get("molecular_weight", 0):.2f} Da'),
        ("Exact mass", f'{p.get("exact_mass", 0):.4f} Da'),
        ("logP",       f'{p.get("logp", 0):.2f}'),
        ("TPSA",       f'{p.get("tpsa", 0):.1f} Å²'),
        ("HBD / HBA",  f'{p.get("num_hbd","?")} / {p.get("num_hba","?")}'),
        ("Heavy atoms",str(p.get("num_heavy_atoms", "—"))),
        ("Rot. bonds", str(p.get("num_rotatable_bonds", "—"))),
    ]
    cells = "".join(
        f'<div class="prop-item"><div class="prop-key">{k}</div><div class="prop-val">{v}</div></div>'
        for k, v in items
    )
    return f'<div class="prop-grid">{cells}</div>'


def _svg_img(svg: str, width: int = 540) -> str:
    b64 = base64.b64encode(svg.encode()).decode()
    return (
        f'<div style="display:flex;justify-content:center;margin:0.5rem 0">'
        f'<div class="mol-frame">'
        f'<img src="data:image/svg+xml;base64,{b64}" width="{width}" />'
        f'</div></div>'
    )


def _build_met_df(mets: List[Dict]) -> pd.DataFrame:
    rows = []
    for m in mets:
        rows.append({
            "SMILES":    m["smiles"],
            "Prob.":     round(m["probability"], 4),
            "Phase":     m["phase"],
            "Reaction":  m.get("reaction_name", "—"),
            "MW (Da)":   f'{m["molecular_weight"]:.2f}' if m.get("molecular_weight") else "—",
            "Formula":   m.get("molecular_formula") or "—",
            "Confidence":m.get("confidence_tier", "—"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Prob.", ascending=False).reset_index(drop=True)
        df.index += 1
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        # Backend status
        ok = st.session_state.backend_ok
        dot = "🟢" if ok is True else ("🔴" if ok is False else "⚪")
        st.markdown(
            f'<div style="font-size:0.78rem;color:var(--slate-400);margin-bottom:0.8rem">'
            f'{dot} Backend: <span style="color:var(--slate-200)">'
            f'{"Online" if ok else ("Offline" if ok is False else "Unknown")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("↻ Ping backend"):
            st.session_state.backend_ok = _check_backend()
            st.rerun()

        st.markdown("---")
        st.markdown('<p class="sec-label">Load example drug</p>', unsafe_allow_html=True)

        for name, (smi, desc) in EXAMPLES.items():
            c1, c2 = st.columns([3, 1])
            c1.markdown(
                f'<span style="font-size:0.85rem;color:var(--white)">{name}</span><br>'
                f'<span style="font-size:0.70rem;color:var(--slate-500)">{desc}</span>',
                unsafe_allow_html=True,
            )
            if c2.button("Load", key=f"ex_{name}"):
                st.session_state.smiles  = smi
                st.session_state.result  = None
                st.session_state.svg     = None
                st.session_state.error   = None
                st.rerun()

        st.markdown("---")
        st.markdown('<p class="sec-label">Prediction parameters</p>', unsafe_allow_html=True)

        p1   = st.slider("Phase I cycles",    1, 3, 1)
        p2   = st.slider("Phase II cycles",   1, 3, 1)
        maxm = st.slider("Max metabolites",   10, 200, 50, 10)
        topn = st.slider("Top soft spots",    1, 10, 3)

        st.markdown("---")
        st.markdown('<p class="sec-label">Highlight colour</p>', unsafe_allow_html=True)
        hr = st.slider("R", 0.0, 1.0, 0.98, 0.05)
        hg = st.slider("G", 0.0, 1.0, 0.25, 0.05)
        hb = st.slider("B", 0.0, 1.0, 0.25, 0.05)
        ha = st.slider("α", 0.0, 1.0, 0.70, 0.05)
        show_scores = st.checkbox("Show atom scores on SVG", value=True)

        st.markdown("---")
        st.session_state.show_all_mets = st.checkbox("Show all metabolites", False)
        debug = st.checkbox("Debug: raw JSON", False)

    return p1, p2, maxm, topn, [hr, hg, hb, ha], show_scores, debug


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _init()
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">MetID</div>'
        '<div class="hero-sub">Metabolite Intelligence Platform · Ensemble Consensus Engine · v2.0</div>'
        '<div class="hero-badges">'
        '<span class="hero-badge">⚗ SyGMa Rule Engine</span>'
        '<span class="hero-badge">🧠 DL Emulator</span>'
        '<span class="hero-badge">🔬 RDKit</span>'
        '<span class="hero-badge">⚡ FastAPI</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    p1, p2, maxm, topn, hcolor, show_scores, debug = _sidebar()

    # ── Input section ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">Molecule input</p>', unsafe_allow_html=True)

    # Try streamlit-ketcher; fall back gracefully to text area
    smiles_from_ketcher = None
    try:
        from streamlit_ketcher import st_ketcher  # type: ignore
        with st.expander("✏️  Draw molecule (Ketcher sketcher)", expanded=False):
            ketcher_val = st_ketcher(
                value=st.session_state.smiles or "",
                height=420,
                key="ketcher_widget",
            )
            if ketcher_val:
                smiles_from_ketcher = ketcher_val
    except ImportError:
        st.markdown(
            '<div class="warn-banner">streamlit-ketcher not installed — '
            'using text input.  Install: <code>pip install streamlit-ketcher</code></div>',
            unsafe_allow_html=True,
        )

    icol, bcol = st.columns([7, 1])
    with icol:
        smiles_input = st.text_area(
            label="SMILES string",
            value=smiles_from_ketcher or st.session_state.smiles,
            height=70,
            placeholder="e.g.  CC(=O)Oc1ccccc1C(=O)O   (Aspirin)",
            label_visibility="collapsed",
            key="smiles_ta",
        )
        st.session_state.smiles = smiles_input

    with bcol:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        run = st.button("▶ Analyse", use_container_width=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run:
        raw = st.session_state.smiles.strip()
        if not raw:
            st.session_state.error = (
                "**Empty input.** Paste a SMILES string or load an example from the sidebar."
            )
        else:
            st.session_state.error = None
            with st.spinner("Running Ensemble Consensus Engine…"):
                result, err = _post_predict(raw, p1, p2, maxm, topn)
            if err:
                st.session_state.error = err
                st.session_state.result = None
                st.session_state.svg = None
            else:
                st.session_state.result = result
                indices = [s["atom_index"] for s in result.get("soft_spots", [])]
                with st.spinner("Rendering molecule SVG…"):
                    canon = result["parent"]["canonical_smiles"]
                    svg, rerr = _post_render(canon, indices, 560, 380,
                                             hcolor, show_scores)
                st.session_state.svg = svg
                if rerr:
                    st.session_state.error = f"SVG render warning: {rerr}"

    # ── Error banner ──────────────────────────────────────────────────────────
    if st.session_state.error:
        st.markdown(
            f'<div class="err-banner">{st.session_state.error}</div>',
            unsafe_allow_html=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.result:
        res    = st.session_state.result
        svg    = st.session_state.svg
        parent = res.get("parent", {})
        mets   = res.get("metabolites", [])
        spots  = res.get("soft_spots", [])
        warns  = res.get("warnings", [])
        stats  = res.get("pipeline_stats", {})

        for w in warns:
            st.markdown(f'<div class="warn-banner">{w}</div>', unsafe_allow_html=True)

        # ── Metric row ────────────────────────────────────────────────────────
        n_consensus = stats.get("consensus_count", 0)
        top_vi      = spots[0].get("vulnerability_index", 0) if spots else 0
        st.markdown(
            f'<div class="metric-row">'
            + _metric(res.get("metabolites_total", 0), "Metabolites", "metric-sky")
            + _metric(n_consensus, "Consensus Matches", "metric-em")
            + _metric(res.get("soft_spots_total", 0), "Soft Spots", "metric-rose")
            + _metric(f"{top_vi:.1f}%", "Peak Vulnerability", "metric-am")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── Three-column layout ───────────────────────────────────────────────
        lcol, mcol, rcol = st.columns([1, 1.3, 1.6], gap="large")

        # ── LEFT: Analytics summary ───────────────────────────────────────────
        with lcol:
            st.markdown('<p class="sec-label">Analytics summary</p>', unsafe_allow_html=True)

            # Pipeline stats card
            st.markdown(
                f'<div class="card" style="margin-bottom:1rem">'
                f'<div style="font-size:0.68rem;color:var(--slate-500);text-transform:uppercase;'
                f'letter-spacing:0.1em;margin-bottom:0.7rem">Pipeline breakdown</div>'
                + "".join([
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:4px 0;border-bottom:1px solid var(--border);font-size:0.8rem">'
                    f'<span style="color:var(--slate-400)">{k}</span>'
                    f'<span style="color:var(--white);font-weight:600">{v}</span></div>'
                    for k, v in [
                        ("SyGMa (Rule-Based)", stats.get("sygma_total", 0)),
                        ("DL Emulator",        stats.get("dl_total", 0)),
                        ("Consensus Verified", stats.get("consensus_count", 0)),
                        ("Rule-Only",          stats.get("rule_only_count", 0)),
                        ("DL-Only",            stats.get("dl_only_count", 0)),
                        ("Elapsed",            f'{res.get("elapsed_s",0):.2f}s'),
                    ]
                ])
                + "</div>",
                unsafe_allow_html=True,
            )

            # Physico-chem props
            st.markdown('<p class="sec-label">Physicochemical properties</p>', unsafe_allow_html=True)
            st.markdown(_prop_grid(parent), unsafe_allow_html=True)

            # Canonical SMILES
            canon = parent.get("canonical_smiles", "")
            if canon:
                st.markdown(
                    f'<div style="margin-top:0.8rem;padding:7px 12px;'
                    f'background:var(--bg-raised);border:1px solid var(--border);'
                    f'border-radius:var(--radius);font-size:0.74rem;'
                    f'color:var(--sky-300);word-break:break-all;">{canon}</div>',
                    unsafe_allow_html=True,
                )

            # Soft spot list
            st.markdown('<p class="sec-label" style="margin-top:1.2rem">Soft spot analysis</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="card">{_spot_rows(spots)}</div>',
                unsafe_allow_html=True,
            )

        # ── MIDDLE: Soft spot map ─────────────────────────────────────────────
        with mcol:
            st.markdown('<p class="sec-label">Soft spot mapping — glowing vulnerability overlay</p>',
                        unsafe_allow_html=True)

            if svg:
                st.markdown(_svg_img(svg, width=520), unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="card empty-state">'
                    '<div class="empty-icon">🔬</div>'
                    '<p>Structure rendering unavailable</p></div>',
                    unsafe_allow_html=True,
                )

            # DL source badge
            dl_src = stats.get("dl_source", "—")
            st.markdown(
                f'<div style="text-align:center;margin-top:0.5rem">'
                f'<span style="font-size:0.68rem;color:var(--slate-500)">DL source: </span>'
                f'<span style="font-size:0.70rem;color:var(--violet-400)">{dl_src}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Engine version
            st.markdown(
                f'<div style="text-align:center;margin-top:0.2rem">'
                f'<span style="font-size:0.65rem;color:var(--slate-500)">'
                f'engine v{res.get("engine_version","—")}</span></div>',
                unsafe_allow_html=True,
            )

        # ── RIGHT: Metabolic tree ─────────────────────────────────────────────
        with rcol:
            st.markdown('<p class="sec-label">Metabolic tree</p>', unsafe_allow_html=True)

            if not mets:
                st.markdown(
                    '<div class="card empty-state">'
                    '<div class="empty-icon">🧪</div>'
                    '<p>No metabolites predicted.<br>'
                    '<span style="font-size:0.76rem">Ensure SyGMa is installed.</span></p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Phase filter
                fc1, fc2, fc3 = st.columns(3)
                if fc1.button("All phases",  key="f_all"): st.session_state.met_filter = "all"
                if fc2.button(f"Phase I ({res.get('phase1_count',0)})", key="f_p1"):
                    st.session_state.met_filter = 1
                if fc3.button(f"Phase II ({res.get('phase2_count',0)})", key="f_p2"):
                    st.session_state.met_filter = 2

                df = _build_met_df(mets)
                flt = st.session_state.get("met_filter", "all")
                if flt != "all":
                    df = df[df["Phase"] == flt]

                limit = None if st.session_state.show_all_mets else 30
                df_show = df if limit is None else df.head(limit)
                if limit and len(df) > limit:
                    st.caption(f"Showing top {limit} of {len(df)} — enable **Show all metabolites** in sidebar")

                # Style
                styled = (
                    df_show.style
                    .format({"Prob.": "{:.4f}"})
                    .background_gradient(subset=["Prob."], cmap="Blues",
                                         vmin=0, vmax=max(df["Prob."].max(), 0.01))
                    .set_properties(**{"font-size": "11px", "font-family": "'JetBrains Mono',monospace"})
                )
                st.dataframe(styled, use_container_width=True,
                             height=min(560, 44 + 36 * len(df_show)))

                # Rich detail expander
                with st.expander("▸ Detailed metabolite cards", expanded=False):
                    for i, row in df_show.iterrows():
                        tier_b  = _badge_tier(row.get("Confidence", ""))
                        phase_b = _badge_phase(int(row["Phase"]))
                        p_str   = f'{row["Prob."]:.4f}'
                        st.markdown(
                            f'<div style="padding:8px 0;border-bottom:1px solid var(--border)">'
                            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
                            f'<span style="font-size:0.68rem;color:var(--slate-500)">#{i}</span>'
                            f'{phase_b} {tier_b}'
                            f'<span style="font-size:0.74rem;color:var(--emerald-400);margin-left:auto">p={p_str}</span>'
                            f'</div>'
                            f'<div style="font-size:0.76rem;color:var(--sky-300);word-break:break-all">{row["SMILES"]}</div>'
                            f'<div style="font-size:0.68rem;color:var(--slate-500);margin-top:2px">{row.get("Reaction","")}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        if debug:
            with st.expander("🔬 Raw JSON (debug)", expanded=False):
                st.json(res)

    elif not st.session_state.error:
        st.markdown(
            '<div class="empty-state" style="margin-top:3rem">'
            '<div class="empty-icon">⚗️</div>'
            '<p style="color:var(--slate-400);font-size:0.9rem">Paste a SMILES string and click '
            '<strong style="color:var(--sky-300)">▶ Analyse</strong></p>'
            '<p style="font-size:0.76rem;color:var(--slate-500);margin-top:0.4rem">'
            'Or load an example compound from the sidebar.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">MetID · Ensemble Consensus Engine v2.0 · '
        'SyGMa · MetaTrans emulator · RDKit · FastAPI · Streamlit<br>'
        'Soft spot scores are computational predictions, not validated clinical data.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
