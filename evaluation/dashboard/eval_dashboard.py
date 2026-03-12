"""
RAG Evaluation Dashboard

Interactive Streamlit dashboard for exploring evaluation results.

Usage:
    streamlit run evaluation/dashboard/eval_dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load latest results
@st.cache_data
def load_results():
    """Load latest evaluation results"""
    results_file = Path("evaluation/results/latest_summary.json")

    if not results_file.exists():
        return None, None

    with open(results_file) as f:
        summary = json.load(f)

    # Try to load detailed results
    detailed_results = None
    results_dir = Path("evaluation/results")

    # Find most recent detailed results file
    result_files = sorted(results_dir.glob("results_*.json"), reverse=True)
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
            detailed_results = data.get("per_query_results", [])

    return summary, detailed_results


def load_history():
    """Load evaluation history"""
    history_file = Path("evaluation/results/evaluation_history.jsonl")

    if not history_file.exists():
        return []

    history = []
    with open(history_file) as f:
        for line in f:
            history.append(json.loads(line))

    return history


st.title("📊 RAG Evaluation Dashboard")
st.caption("Comprehensive evaluation analysis and error exploration")

# Load data
summary, detailed_results = load_results()

if summary is None:
    st.warning("⚠️ No evaluation results found. Run evaluation first:")
    st.code("python evaluation/pipelines/run_evaluation.py")
    st.stop()

# Sidebar with run info
with st.sidebar:
    st.header("Evaluation Info")
    st.metric("Dataset", Path(summary["dataset"]).name)
    st.metric("Total Queries", summary["total_queries"])
    st.metric("Corpus Size", summary.get("corpus_size", "Unknown"))

    st.divider()

    st.subheader("Quick Stats")
    st.metric("Success Rate", f"{summary['overall_accuracy']:.1%}")
    st.metric("Avg Precision@3", f"{summary['avg_precision_at_3']:.3f}")
    st.metric("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Query Explorer", "💥 Failure Analysis"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("Evaluation Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Accuracy",
            f"{summary['overall_accuracy']:.1%}",
            delta=None
        )

    with col2:
        st.metric(
            "Avg Precision@3",
            f"{summary['avg_precision_at_3']:.3f}",
            delta=None
        )

    with col3:
        st.metric(
            "Successful Queries",
            f"{summary['successful']}/{summary['total_queries']}",
            delta=None
        )

    with col4:
        st.metric(
            "Failed Queries",
            f"{summary['failed']}/{summary['total_queries']}",
            delta=None if summary['failed'] == 0 else f"-{summary['failed']}"
        )

    st.divider()

    # Performance by difficulty
    if "by_difficulty" in summary and summary["by_difficulty"]:
        st.subheader("Performance by Difficulty")

        diff_data = []
        for difficulty, stats in summary["by_difficulty"].items():
            diff_data.append({
                "Difficulty": difficulty.capitalize(),
                "Queries": stats["count"],
                "Success Rate": f"{stats['success_rate']:.1%}",
                "Avg P@3": f"{stats['avg_precision_at_3']:.3f}"
            })

        df_diff = pd.DataFrame(diff_data)
        st.dataframe(df_diff, use_container_width=True, hide_index=True)

        # Bar chart
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d["Difficulty"] for d in diff_data],
            y=[float(d["Success Rate"].strip('%')) for d in diff_data],
            name="Success Rate (%)",
            marker_color="lightblue"
        ))

        fig.update_layout(
            title="Success Rate by Difficulty",
            xaxis_title="Difficulty",
            yaxis_title="Success Rate (%)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Latency info
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Latency", f"{summary['avg_latency_ms']:.0f}ms")
        st.caption("Average query processing time")

    with col2:
        # Estimate cost (rough)
        queries = summary['total_queries']
        est_cost = queries * 0.005  # Rough estimate
        st.metric("Estimated Cost", f"${est_cost:.4f}")
        st.caption(f"For {queries} queries (~$0.005/query estimate)")


# ==================== TAB 2: QUERY EXPLORER ====================
with tab2:
    st.header("Query-Level Exploration")

    if not detailed_results:
        st.warning("Detailed results not available")
        st.stop()

    # Create DataFrame
    df_queries = pd.DataFrame([
        {
            "ID": r["query_id"],
            "Question": r["question"][:80] + "..." if len(r["question"]) > 80 else r["question"],
            "Difficulty": r.get("difficulty", "unknown"),
            "Category": r.get("category", "unknown"),
            "P@3": f"{r['metrics']['precision_at_3']:.3f}",
            "Latency (ms)": f"{r['latency_ms']:.0f}",
            "Status": "✅ Pass" if r["success"] else "❌ Fail"
        }
        for r in detailed_results
    ])

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Pass", "Fail"]
        )

    with col2:
        difficulties = ["All"] + list(df_queries["Difficulty"].unique())
        difficulty_filter = st.selectbox(
            "Filter by Difficulty",
            difficulties
        )

    with col3:
        categories = ["All"] + list(df_queries["Category"].unique())
        category_filter = st.selectbox(
            "Filter by Category",
            categories
        )

    # Apply filters
    filtered_df = df_queries.copy()

    if status_filter != "All":
        status_icon = "✅ Pass" if status_filter == "Pass" else "❌ Fail"
        filtered_df = filtered_df[filtered_df["Status"] == status_icon]

    if difficulty_filter != "All":
        filtered_df = filtered_df[filtered_df["Difficulty"] == difficulty_filter]

    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["Category"] == category_filter]

    # Display table
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.divider()

    # Detailed view for selected query
    st.subheader("Query Details")

    selected_id = st.selectbox(
        "Select Query ID",
        [r["query_id"] for r in detailed_results]
    )

    # Find selected result
    selected = next(r for r in detailed_results if r["query_id"] == selected_id)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Question:**")
        st.info(selected["question"])

    with col2:
        st.metric("Precision@3", f"{selected['metrics']['precision_at_3']:.3f}")
        st.metric("Latency", f"{selected['latency_ms']:.0f}ms")

    # Expected vs Retrieved
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Expected Sources:**")
        for src in selected["expected_sources"]:
            st.write(f"- {src}")

    with col2:
        st.markdown("**Retrieved Sources:**")
        for src in selected["retrieved_filenames"]:
            is_correct = src in selected["expected_sources"]
            icon = "✅" if is_correct else "❌"
            st.write(f"{icon} {src}")

    # Answer
    if selected.get("answer"):
        st.markdown("**Generated Answer:**")
        st.success(selected["answer"])

    # Verification
    if selected.get("verification"):
        st.markdown("**Verification:**")
        verdict = selected["verification"].get("verdict", "UNKNOWN")
        reason = selected["verification"].get("reason", "No reason")

        if verdict == "SUPPORTED":
            st.success(f"**{verdict}**: {reason}")
        elif verdict == "PARTIALLY_SUPPORTED":
            st.warning(f"**{verdict}**: {reason}")
        else:
            st.error(f"**{verdict}**: {reason}")


# ==================== TAB 3: FAILURE ANALYSIS ====================
with tab3:
    st.header("Failure Analysis")

    if not detailed_results:
        st.warning("Detailed results not available")
        st.stop()

    # Get failures
    failures = [r for r in detailed_results if not r["success"]]

    if not failures:
        st.success("🎉 No failures! All queries passed.")
        st.stop()

    st.metric("Total Failures", len(failures))

    # Run failure analysis
    from evaluation.analysis.failure_analyzer import FailureAnalyzer

    analyzer = FailureAnalyzer()

    # Load dataset to get ground truth
    dataset_path = Path("evaluation/datasets/base_eval.json")
    with open(dataset_path) as f:
        dataset = json.load(f)

    query_items = {q["id"]: q for q in dataset["queries"]}

    # Analyze each failure
    for result in failures:
        query_item = query_items.get(result["query_id"], {})
        analyzer.analyze_failure(query_item, result)

    # Display failure mode distribution
    st.subheader("Failure Mode Distribution")

    failure_mode_counts = {}
    for analysis in analyzer.analyses:
        for mode in analysis["failure_modes"]:
            mode_name = mode.value
            failure_mode_counts[mode_name] = failure_mode_counts.get(mode_name, 0) + 1

    if failure_mode_counts:
        # Create pie chart
        import plotly.express as px

        df_modes = pd.DataFrame([
            {"Mode": mode, "Count": count}
            for mode, count in failure_mode_counts.items()
        ])

        fig = px.pie(
            df_modes,
            values="Count",
            names="Mode",
            title="Failure Modes"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Root cause distribution
    st.subheader("Root Cause Distribution")

    root_causes = {}
    for analysis in analyzer.analyses:
        rc = analysis["root_cause"]
        root_causes[rc] = root_causes.get(rc, 0) + 1

    df_causes = pd.DataFrame([
        {"Root Cause": cause.replace("_", " ").title(), "Count": count}
        for cause, count in root_causes.items()
    ])

    st.dataframe(df_causes, use_container_width=True, hide_index=True)

    st.divider()

    # Individual failure details
    st.subheader("Detailed Failure Reports")

    for analysis in analyzer.analyses:
        severity_color = {
            "critical": "🔴",
            "high": "🟠",
            "moderate": "🟡",
            "low": "🟢"
        }

        icon = severity_color.get(analysis["severity"], "⚪")

        with st.expander(f"{icon} {analysis['query_id']}: {analysis['question'][:60]}..."):
            st.markdown(f"**Severity:** {analysis['severity'].upper()}")
            st.markdown(f"**Root Cause:** {analysis['root_cause'].replace('_', ' ').title()}")

            st.markdown("**Failure Modes:**")
            for mode in analysis["failure_modes"]:
                st.write(f"- {mode.value}")

            st.markdown("**Explanation:**")
            st.info(analysis["explanation"])

            st.markdown("**Recommended Fix:**")
            st.code(analysis["recommended_fix"])

    # Generate and download report
    st.divider()
    st.subheader("Export Report")

    if st.button("Generate Markdown Report"):
        report = analyzer.generate_report()
        st.download_button(
            label="Download Report",
            data=report,
            file_name="failure_analysis_report.md",
            mime="text/markdown"
        )
        st.success("Report generated! Click download button above.")


# Footer
st.divider()
st.caption(f"Last updated: {summary.get('timestamp', 'Unknown')}")
