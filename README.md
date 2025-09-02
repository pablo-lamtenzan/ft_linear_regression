# Linear Regression

Predicts car prices based on mileage using gradient descent linear regression.

## Requirements

**Python 3.9+** is required.

### Install Python

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

**macOS (with Homebrew):**
```bash
brew install python@3.9
```

## Quick Start

1. **Install uv** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

3. **Train a model**:
   ```bash
   uv run train data.csv
   ```

4. **Make predictions**:
   ```bash
   uv run predict 150000
   ```

5. **Launch interactive dashboard**:
   ```bash
   uv run invoke web
   ```

## How It Works

The project implements linear regression with gradient descent to learn the relationship:
```
price = θ₀ + θ₁ × mileage
```

- **Training**: Reads CSV data (km,price), normalizes it, and uses gradient descent to find optimal θ₀ and θ₁ parameters
- **Prediction**: Loads trained parameters and predicts prices for new mileage values

## Available Commands (via invoke)

```bash
uv run invoke --list
```

**Development tasks:**
- `uv run invoke test` - Run tests
- `uv run invoke lint` - Check code style
- `uv run invoke format` - Format code
- `uv run invoke type-check` - Run type checking
- `uv run invoke check` - Run all checks

**Model tasks:**
- `uv run invoke train [data-file]` - Train model
- `uv run invoke predict <mileage>` - Predict price
- `uv run invoke web` - Launch interactive web dashboard

## 🚀 Usage

### 🐳 Docker (Recommended - Zero Setup)

The easiest way to run the application:

```bash
# Using docker-compose (recommended)
docker-compose up --build

# Or using Docker directly
docker build -t linear-regression-app .
docker run -p 8501:8501 linear-regression-app
```

Access the dashboard at `http://localhost:8501`

### 💻 Local Development

## CLI Options

**Training:**
```bash
uv run train data.csv --learning-rate 0.01 --max-epochs 1000 --precision
```

**Prediction:**
```bash
uv run predict 150000 --verbose --check-range
```

## Interactive Dashboard

Launch the web-based dashboard for an enhanced experience:

```bash
uv run invoke web
```

## 🐳 Docker Deployment

For the easiest setup, use Docker to run the application without installing any dependencies:

### Quick Start with Docker

```bash
# Build and run with docker-compose (recommended)
docker-compose up --build

# Or build and run manually
docker build -t linear-regression-app .
docker run -p 8501:8501 linear-regression-app
```

The application will be available at `http://localhost:8501`

### Docker Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+ (optional, for docker-compose method)

### Docker Features

- ✅ **Zero setup** - No need to install Python, uv, or dependencies
- ✅ **Consistent environment** - Same runtime across all systems
- ✅ **Health checks** - Automatic container health monitoring
- ✅ **Security** - Runs as non-root user
- ✅ **Production ready** - Optimized for deployment
