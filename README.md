# Graph RAG with Kuzu, DSPy and marimo

Source code for course project to build a Graph RAG with Kuzu, [DSPy](https://dspy.ai/) and [marimo](https://docs.marimo.io/) (open source, reactive notebooks for Python).

## Setup

We recommend using the `uv` package manager
to manage dependencies.

```bash
# Uses the local pyproject.toml to add dependencies
uv sync

# Don't forget to source virtual env
source .venv/bin/activate
```

### Start Kuzu Database
```bash
docker compose up
```
Go to `localhost:8000` you can check the UI of the database

### Enrich the graph 
Create the required graph in Kuzu using the following script:

```bash
uv run create_nobel_api_graph.py
```

#### Run as a notebook

To run Graph RAG pipeline as an marimo notebook, run marimo as follows:
```bash
# Open a marimo notebook in edit mode
marimo edit submission_workflow.py
```

#### Run as an app

To run Graph RAG pipeline as an app, use the `streamlit run` command.

```bash
streamlit run main.py
```