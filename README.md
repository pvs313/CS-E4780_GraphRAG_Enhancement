# Graph RAG Project

Built on top of the provided source code and has an improved performance and accuracy. 

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
marimo edit submission_workflow.py
```

#### Run as an app

To run Graph RAG pipeline as an app, use the `streamlit run` command.

```bash
streamlit run main.py
```