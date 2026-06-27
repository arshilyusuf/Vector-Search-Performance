"""
Streamlit Dashboard — Vector Search Performance Profiler.

Connects to the FastAPI backend, submits benchmark jobs, tracks
progress in real-time, and renders interactive Plotly charts.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vector Search Profiler",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

API_BASE = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Root URL of the running FastAPI backend.",
)

POLL_INTERVAL_S = 2  # seconds between status polls
MAX_POLL_ATTEMPTS = 300  # 10 minutes timeout

INDEX_COLOR_MAP = {
    "IndexFlatL2": "#4C72B0",
    "IndexIVFFlat": "#DD8452",
    "IndexHNSW": "#55A868",
}


def _get(path: str) -> Dict[str, Any]:
    """Make a GET request to the API.

    Args:
        path: API path relative to API_BASE.

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
    """
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{API_BASE}{path}")
        resp.raise_for_status()
        return resp.json()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make a POST request to the API.

    Args:
        path: API path relative to API_BASE.
        payload: JSON body.

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
    """
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{API_BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Benchmark Config")

n_vectors = st.sidebar.select_slider(
    "Base vectors (n)",
    options=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    value=50_000,
)

n_queries = st.sidebar.slider("Query vectors", 100, 5_000, 1_000, step=100)
dimension = st.sidebar.selectbox("Dimension (d)", [64, 128, 256, 512], index=1)

available_indexes = ["IndexFlatL2", "IndexIVFFlat", "IndexHNSW"]
selected_indexes = st.sidebar.multiselect(
    "Index types to benchmark",
    available_indexes,
    default=["IndexIVFFlat", "IndexHNSW"],
)

k_values_input = st.sidebar.text_input("K values (comma-separated)", "1,10,100")
nlist = st.sidebar.slider("IVF nlist", 10, 500, 100)
nprobe = st.sidebar.slider("IVF nprobe", 1, 100, 10)
hnsw_m = st.sidebar.slider("HNSW M", 8, 128, 32)

random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("🔍 Vector Search Performance Profiler")
st.caption("FAISS benchmark suite — IndexFlatL2 · IndexIVFFlat · IndexHNSW")

col_start, col_health = st.columns([4, 1])

with col_health:
    try:
        health = _get("/health")
        st.success(f"API: {health.get('status', 'unknown').upper()}")
    except Exception:
        st.error("API: OFFLINE")

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

with col_start:
    run_clicked = st.button(
        "▶  Run Benchmark",
        type="primary",
        use_container_width=True,
        disabled=len(selected_indexes) == 0,
    )

if run_clicked:
    try:
        k_values = [int(k.strip()) for k in k_values_input.split(",") if k.strip()]
        payload = {
            "n_vectors": n_vectors,
            "n_queries": n_queries,
            "dimension": dimension,
            "index_types": selected_indexes,
            "k_values": k_values,
            "nlist": nlist,
            "nprobe": nprobe,
            "hnsw_m": hnsw_m,
            "random_seed": int(random_seed),
        }
        response = _post("/run-benchmark", payload)
        st.session_state["job_id"] = response["job_id"]
        st.session_state["report"] = None
        st.session_state["status"] = "queued"
        st.toast("Benchmark queued!", icon="🚀")
    except Exception as exc:
        st.error(f"Failed to start benchmark: {exc}")

# ---------------------------------------------------------------------------
# Status polling + progress bar
# ---------------------------------------------------------------------------

job_id: Optional[str] = st.session_state.get("job_id")

if job_id and st.session_state.get("status") not in ("completed", "failed"):
    st.markdown("---")
    st.subheader("⏳ Benchmark in progress…")

    status_ph = st.empty()
    progress_ph = st.progress(0, text="Waiting for worker…")

    for attempt in range(MAX_POLL_ATTEMPTS):
        try:
            status_data = _get(f"/get-status/{job_id}")
            current_status = status_data["status"]
        except Exception as exc:
            status_ph.error(f"Polling error: {exc}")
            break

        st.session_state["status"] = current_status
        progress_frac = min(0.95, attempt / MAX_POLL_ATTEMPTS)

        if current_status == "queued":
            progress_ph.progress(0.05, text="Job queued — waiting for worker…")
        elif current_status == "running":
            progress_ph.progress(progress_frac, text="Running FAISS benchmarks…")
        elif current_status == "completed":
            progress_ph.progress(1.0, text="Done!")
            status_ph.success("✅ Benchmark completed!")
            time.sleep(0.5)

            try:
                report_data = _get(f"/get-report/{job_id}")
                st.session_state["report"] = report_data["report"]
            except Exception as exc:
                st.error(f"Failed to fetch report: {exc}")

            st.rerun()
        elif current_status == "failed":
            progress_ph.progress(0, text="Failed.")
            status_ph.error(
                f"❌ Benchmark failed: {status_data.get('error', 'unknown error')}"
            )
            break

        time.sleep(POLL_INTERVAL_S)

# ---------------------------------------------------------------------------
# Results section
# ---------------------------------------------------------------------------

report: Optional[Dict] = st.session_state.get("report")

if report:
    st.markdown("---")
    st.subheader("📊 Results")

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Base Vectors", f"{report['total_vectors']:,}")
    meta_col2.metric("Query Vectors", f"{report['total_queries']:,}")
    meta_col3.metric(
        "GT Build Time",
        f"{report['ground_truth_build_time_s']:.2f}s",
    )

    # ── Build flat DataFrame ───────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    for idx_name, k_map in report["results"].items():
        for k_str, m in k_map.items():
            rows.append(
                {
                    "Index": idx_name,
                    "K": int(k_str),
                    "Latency Mean (ms)": round(m["latency_mean_ms"], 4),
                    "Latency p50 (ms)": round(m["latency_p50_ms"], 4),
                    "Latency p95 (ms)": round(m["latency_p95_ms"], 4),
                    "Latency p99 (ms)": round(m["latency_p99_ms"], 4),
                    "Recall@K": round(m["recall"], 4),
                    "MRR": round(m["mrr"], 4),
                    "Build Time (s)": round(m.get("build_time_s") or 0, 3),
                }
            )

    df = pd.DataFrame(rows)

    # ── K selector ────────────────────────────────────────────────────
    available_ks = sorted(df["K"].unique().tolist())
    selected_k = st.selectbox(
        "Select K for charts", available_ks, index=min(1, len(available_ks) - 1)
    )
    idx_filter = st.multiselect(
        "Show index types",
        df["Index"].unique().tolist(),
        default=df["Index"].unique().tolist(),
    )

    df_filtered = df[(df["K"] == selected_k) & (df["Index"].isin(idx_filter))]

    chart_col1, chart_col2 = st.columns(2)

    # ── Chart 1: Latency vs Recall scatter ────────────────────────────
    with chart_col1:
        fig_scatter = go.Figure()
        for idx_name in df_filtered["Index"].unique():
            row = df_filtered[df_filtered["Index"] == idx_name].iloc[0]
            fig_scatter.add_trace(
                go.Scatter(
                    x=[row["Latency Mean (ms)"]],
                    y=[row["Recall@K"]],
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color=INDEX_COLOR_MAP.get(idx_name, "#888"),
                        line=dict(width=2, color="white"),
                    ),
                    text=[idx_name.replace("Index", "")],
                    textposition="top center",
                    name=idx_name,
                )
            )
        fig_scatter.update_layout(
            title=f"Latency vs Recall@{selected_k}",
            xaxis_title="Mean Latency per Query (ms)",
            yaxis_title=f"Recall@{selected_k}",
            yaxis=dict(range=[0, 1.05]),
            legend_title="Index",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Chart 2: Build times bar ───────────────────────────────────────
    with chart_col2:
        build_df = df[df["Index"].isin(idx_filter)].drop_duplicates("Index")[
            ["Index", "Build Time (s)"]
        ]
        fig_build = px.bar(
            build_df,
            x="Index",
            y="Build Time (s)",
            color="Index",
            color_discrete_map=INDEX_COLOR_MAP,
            title="Index Build Time",
            text_auto=".2f",
            height=400,
            template="plotly_white",
        )
        fig_build.update_traces(textposition="outside")
        fig_build.update_layout(showlegend=False)
        st.plotly_chart(fig_build, use_container_width=True)

    # ── Chart 3: Latency percentiles grouped bar ───────────────────────
    st.subheader("Latency Distribution (p50 / p95 / p99)")
    melt_cols = ["Index", "Latency p50 (ms)", "Latency p95 (ms)", "Latency p99 (ms)"]
    df_melted = df_filtered[melt_cols].melt(
        id_vars="Index", var_name="Percentile", value_name="Latency (ms)"
    )
    fig_lat = px.bar(
        df_melted,
        x="Index",
        y="Latency (ms)",
        color="Percentile",
        barmode="group",
        title=f"Latency Percentiles @ K={selected_k}",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_lat, use_container_width=True)

    # ── Raw data table ─────────────────────────────────────────────────
    with st.expander("📋 Raw metrics table"):
        st.dataframe(
            df.style.format(
                {
                    "Latency Mean (ms)": "{:.4f}",
                    "Latency p50 (ms)": "{:.4f}",
                    "Latency p95 (ms)": "{:.4f}",
                    "Latency p99 (ms)": "{:.4f}",
                    "Recall@K": "{:.4f}",
                    "MRR": "{:.4f}",
                }
            ),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            file_name="benchmark_results.csv",
            mime="text/csv",
        )

elif job_id is None:
    st.info(
        "👈 Configure parameters in the sidebar and click **Run Benchmark** to start."
    )
