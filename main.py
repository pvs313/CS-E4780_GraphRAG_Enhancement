import os
import sys
import io
import contextlib
from typing import Optional

import streamlit as st
import dspy
from dotenv import load_dotenv

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    database,
    models,
    cache,
    workflow,
    evaluation,
)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="App to test Graph RAG Worflow",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def init_dspy():
    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        st.error("Please set API_KEY environment variable")
        st.stop()
    
    lm = dspy.LM(
        model="openrouter/google/gemini-2.0-flash-001",
        api_base="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )
    dspy.configure(lm=lm)
    return lm

@st.cache_resource
def init_db():
    return database.get_db_connection("nobel.kuzu", read_only=True)


@st.cache_resource
def init_caches():
    return cache.LRUCache(maxsize=128), cache.LRUCache(maxsize=128)

init_dspy()
conn = init_db()
text2cypher_cache, prune_schema_cache = init_caches()

st.sidebar.title("⚙️ Settings")
page = st.sidebar.radio(
    "Navigation",
    ["Ask Question", "Benchmarking", "Accuracy Evaluation"]
)

if page == "Ask Question":
    st.title("🔍 Ask questions using Graph RAG")
    
    st.markdown("""
    Ask questions about the Nobel Prize dataset.
    """)
    
    st.sidebar.subheader("Cache Settings")
    use_cache = st.sidebar.checkbox("Use Text2Cypher Cache", value=True)
    use_prune_cache = st.sidebar.checkbox("Use Schema Prune Cache", value=True)
    use_full_schema_cache = st.sidebar.checkbox("Use Full Schema Cache", value=True)
    
    question = st.text_input(
        "Enter your question:",
        value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        help="Ask a question about Nobel Prize winners"
    )
    
    if st.button("Ask Question", type="primary"):
        if not question:
            st.warning("Please enter a question")
        else:
            output_capture = io.StringIO()
            
            with st.spinner("Processing question..."):
                with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                    query, context, answer = workflow.run_full_workflow(
                        conn=conn,
                        question=question,
                        text2cypher_cache=text2cypher_cache,
                        prune_schema_cache=prune_schema_cache,
                        use_cache=use_cache,
                        use_prune_cache=use_prune_cache,
                        use_full_schema_cache=use_full_schema_cache,
                        return_timings=False,
                    )
            
            output_text = output_capture.getvalue()
            
            st.subheader("📝 Answer")
            if answer:
                st.success(getattr(answer, "response", "No answer generated"))
            else:
                st.warning("No answer was generated. Check the intermediate stages below.")
            
            with st.expander("🔍 Intermediate Stages", expanded=True):
                st.text(output_text)
            
            with st.expander("📊 Generated Cypher Query"):
                st.code(query, language="cypher")
            
            if context:
                with st.expander("📦 Query Results"):
                    st.json(context if isinstance(context, list) else [context])
            
            if st.button("Clear All Caches"):
                text2cypher_cache.clear()
                prune_schema_cache.clear()
                database.clear_full_schema_cache()
                st.success("All caches cleared!")
                st.rerun()

elif page == "Benchmarking":
    st.title("⚡ Performance Benchmarking")
    
    st.markdown("""
    Run benchmarks to compare performance with cache ON vs cache OFF.
    The benchmark runs each scenario twice and reports timings from the second run.
    """)
    
    test_questions = evaluation.load_test_questions()
    
    st.sidebar.subheader("Benchmark Settings")
    num_questions = st.sidebar.slider(
        "Number of questions",
        min_value=1,
        max_value=min(10, len(test_questions)),
        value=min(10, len(test_questions))
    )
    
    selected_questions = [item["question"] for item in test_questions[:num_questions]]
    
    st.subheader("📋 Test Questions")
    for i, q in enumerate(selected_questions, 1):
        st.write(f"{i}. {q}")
    
    if st.button("Run Benchmark", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        output_capture = io.StringIO()
        
        status_text.text("Running Cache OFF scenario...")
        database.clear_full_schema_cache()
        text2cypher_cache.clear()
        prune_schema_cache.clear()
        
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            cache_off_run1 = evaluation.run_benchmark(
                conn, selected_questions, text2cypher_cache, prune_schema_cache,
                False, False, False, "Cache OFF run 1"
            )
            progress_bar.progress(25)
            
            cache_off_run2 = evaluation.run_benchmark(
                conn, selected_questions, text2cypher_cache, prune_schema_cache,
                False, False, False, "Cache OFF run 2"
            )
            progress_bar.progress(50)
        
        cache_off_samples, cache_off_avg = evaluation.summarize_second_run(cache_off_run2)
        
        status_text.text("Running Cache ON scenario...")
        database.clear_full_schema_cache()
        text2cypher_cache.clear()
        prune_schema_cache.clear()
        
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            cache_on_run1 = evaluation.run_benchmark(
                conn, selected_questions, text2cypher_cache, prune_schema_cache,
                True, True, True, "Cache ON run 1"
            )
            progress_bar.progress(75)
            
            cache_on_run2 = evaluation.run_benchmark(
                conn, selected_questions, text2cypher_cache, prune_schema_cache,
                True, True, True, "Cache ON run 2"
            )
            progress_bar.progress(100)
        
        cache_on_samples, cache_on_avg = evaluation.summarize_second_run(cache_on_run2)
        
        status_text.text("Benchmark complete!")
        
        st.subheader("📊 Benchmark Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cache OFF Avg Total", f"{cache_off_avg.get('total', 0):.2f} ms")
        
        with col2:
            st.metric("Cache ON Avg Total", f"{cache_on_avg.get('total', 0):.2f} ms")
        
        total_off = cache_off_avg.get("total", 0)
        total_on = cache_on_avg.get("total", 0)
        if total_off and total_on:
            speedup = total_off / total_on
            st.metric("Speedup", f"{speedup:.2f}x")
        
        st.subheader("Stage Breakdown (Second Runs)")
        
        stages = [s for s in cache_off_avg.keys() if s != "total"]
        stages.sort(key=lambda s: cache_off_avg.get(s, 0), reverse=True)
        
        results_data = []
        for stage in stages + ["total"]:
            off_ms = cache_off_avg.get(stage, 0.0)
            on_ms = cache_on_avg.get(stage, 0.0)
            saved = off_ms - on_ms
            pct = (saved / off_ms * 100) if off_ms else 0.0
            results_data.append({
                "Stage": stage if stage != "total" else "TOTAL",
                "Cache OFF (ms)": f"{off_ms:.2f}",
                "Cache ON (ms)": f"{on_ms:.2f}",
                "Time Saved (ms)": f"{saved:.2f}",
                "Savings (%)": f"{pct:.1f}%"
            })
        
        st.dataframe(results_data, use_container_width=True)
        
        # Display full output
        with st.expander("📋 Full Benchmark Output"):
            st.text(output_capture.getvalue())

elif page == "Accuracy Evaluation":
    st.title("✅ Accuracy Evaluation")
    
    st.markdown("""
    Evaluate the accuracy of the Graph RAG workflow on test questions.
    Compare full mode (with exemplars, postprocessing, refinement) vs simple mode.
    """)
    
    test_questions = evaluation.load_test_questions()
    
    st.sidebar.subheader("Evaluation Settings")
    num_questions = st.sidebar.slider(
        "Number of questions",
        min_value=1,
        max_value=min(10, len(test_questions)),
        value=min(10, len(test_questions))
    )
    
    selected_test_questions = test_questions[:num_questions]
    
    st.subheader("📋 Test Questions")
    for i, item in enumerate(selected_test_questions, 1):
        st.write(f"{i}. {item['question']}")
    
    if st.button("Run Accuracy Evaluation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        output_capture = io.StringIO()
        
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            results = evaluation.evaluate_accuracy_comparison(
                conn=conn,
                test_questions=selected_test_questions,
                text2cypher_cache=text2cypher_cache,
                prune_schema_cache=prune_schema_cache,
                use_cache=False,
                use_prune_cache=False,
            )
        
        status_text.text("Evaluation complete!")
        
        st.subheader("📊 Accuracy Results - Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Full Mode Accuracy", f"{results['full_mode']['average_accuracy']:.2%}")
        with col2:
            st.metric("Simple Mode Accuracy", f"{results['simple_mode']['average_accuracy']:.2%}")
        with col3:
            diff = results['comparison']['average_accuracy_difference']
            st.metric("Difference", f"{diff:+.2%}")
        with col4:
            st.metric("Total Questions", results['comparison']['total_questions'])
        
        st.subheader("Summary")
        summary_data = {
            "Mode": ["Full Mode", "Simple Mode"],
            "Avg Accuracy": [
                f"{results['full_mode']['average_accuracy']:.2%}",
                f"{results['simple_mode']['average_accuracy']:.2%}"
            ],
            "Any Match": [
                f"{results['full_mode']['contains_any_count']}/{results['comparison']['total_questions']}",
                f"{results['simple_mode']['contains_any_count']}/{results['comparison']['total_questions']}"
            ],
            "All Matches": [
                f"{results['full_mode']['contains_all_count']}/{results['comparison']['total_questions']}",
                f"{results['simple_mode']['contains_all_count']}/{results['comparison']['total_questions']}"
            ]
        }
        st.dataframe(summary_data, use_container_width=True)
        
        st.subheader("Detailed Results - Full Mode")
        results_data_full = []
        for r in results["full_mode"]["results"]:
            acc = r["accuracy_result"]
            results_data_full.append({
                "Question": r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"],
                "Matched": f"{acc['match_count']}/{acc['total_expected']}",
                "Accuracy": f"{acc['accuracy']:.2%}",
                "Contains All": "✓" if acc["contains_all"] else "✗"
            })
        st.dataframe(results_data_full, use_container_width=True)
        
        st.subheader("Detailed Results - Simple Mode")
        results_data_simple = []
        for r in results["simple_mode"]["results"]:
            acc = r["accuracy_result"]
            results_data_simple.append({
                "Question": r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"],
                "Matched": f"{acc['match_count']}/{acc['total_expected']}",
                "Accuracy": f"{acc['accuracy']:.2%}",
                "Contains All": "✓" if acc["contains_all"] else "✗"
            })
        st.dataframe(results_data_simple, use_container_width=True)
        
        with st.expander("📋 Full Evaluation Output"):
            st.text(output_capture.getvalue())

