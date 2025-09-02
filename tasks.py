"""Invoke tasks for project automation."""

from invoke import task


@task
def lint(c):
    """Run ruff linter."""
    c.run("uv run ruff check .")


@task
def format(c):
    """Run ruff formatter."""
    c.run("uv run ruff format .")


@task
def type_check(c):
    """Run mypy type checker."""
    c.run("uv run mypy src")


@task
def test(c):
    """Run pytest tests."""
    c.run("uv run pytest")


@task
def test_coverage(c):
    """Run pytest with coverage."""
    c.run("uv run pytest --cov=src --cov-report=term-missing")


@task
def install(c):
    """Install dependencies."""
    c.run("uv sync --dev")


@task
def clean(c):
    """Clean build artifacts."""
    c.run("rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage")
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +", warn=True)


@task(lint, format, type_check, test)
def check(c):
    """Run all checks (lint, format, type check, test)."""
    print("All checks completed!")


@task
def train(c, data_file="data.csv"):
    """Train the linear regression model."""
    c.run(f"uv run train {data_file}")


@task
def predict(c, mileage):
    """Predict car price for given mileage."""
    c.run(f"uv run predict {mileage}")


@task
def web(c):
    """Launch the interactive Streamlit web dashboard."""
    print("🚀 Starting Streamlit dashboard...")
    print("📱 Open your browser to: http://localhost:8501")
    c.run("uv run streamlit run streamlit_app.py")
