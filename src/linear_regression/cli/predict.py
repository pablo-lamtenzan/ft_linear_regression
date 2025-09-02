"""Prediction program CLI for linear regression."""

import sys
from pathlib import Path
from typing import Union

import click

from ..data.exceptions import InvalidArgException, InvalidCsvException
from ..services.prediction import PredictionService


@click.command()
@click.argument("mileage", type=float)
@click.option(
    "--model-file",
    "-m",
    default="model.json",
    type=str,
    help="Path to trained model file (default: model.json)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed prediction information"
)
@click.option(
    "--check-range", is_flag=True, help="Check if mileage is within training data range"
)
@click.option(
    "--batch-file",
    type=click.Path(exists=True),
    help="File containing multiple mileages to predict (one per line)",
)
def main(
    mileage: float,
    model_file: str,
    verbose: bool,
    check_range: bool,
    batch_file: Union[Path, None],
) -> None:
    """
    Predict car price based on mileage using trained linear regression model.

    MILEAGE: Car mileage in kilometers

    Example:
        predict 150000
        predict 150000 --model-file my_model.json --verbose
    """
    try:
        # Initialize prediction service
        prediction_service = PredictionService(model_file)

        # Load the trained model
        try:
            prediction_service.load_model()
        except InvalidCsvException as e:
            click.echo(f"Model error: {e}", err=True)
            click.echo("Hint: Train a model first using the training program", err=True)
            sys.exit(1)

        if verbose:
            click.echo("Model loaded successfully!")

            # Show model info
            model_info = prediction_service.get_model_info()
            click.echo(f"Model file: {model_info['model_file']}")
            click.echo(
                f"Parameters: θ₀={model_info['theta0']:.6f}, θ₁={model_info['theta1']:.6f}"
            )

            if model_info["scaler"]:
                scaler = model_info["scaler"]
                click.echo(
                    f"Training data range: {scaler['km_min']:,.0f} - {scaler['km_max']:,.0f} km"
                )
            click.echo()

        # Handle batch prediction if file provided
        if batch_file:
            try:
                with open(batch_file) as f:
                    mileages = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith(
                            "#"
                        ):  # Skip empty lines and comments
                            try:
                                mile = float(line)
                                mileages.append(mile)
                            except ValueError:
                                click.echo(
                                    f"Warning: Invalid mileage on line {line_num}: {line}",
                                    err=True,
                                )

                if not mileages:
                    click.echo("Error: No valid mileages found in batch file", err=True)
                    sys.exit(1)

                if verbose:
                    click.echo(
                        f"Processing {len(mileages)} mileages from batch file..."
                    )
                    click.echo()

                # Make batch predictions
                results = prediction_service.predict_batch(mileages, verbose=verbose)

                if not verbose:
                    # Simple output format for batch processing
                    for mileage, price in results:
                        click.echo(f"{mileage:,.0f},{price:.2f}")

            except Exception as e:
                click.echo(f"Error processing batch file: {e}", err=True)
                sys.exit(1)

        else:
            # Single prediction
            # Check range if requested
            if check_range:
                validation = prediction_service.validate_prediction_range(mileage)
                if not validation["within_range"] and validation["warning"]:
                    click.echo(f"Warning: {validation['warning']}", err=True)
                    if not verbose:
                        click.echo("Use --verbose for more details", err=True)

            # Make prediction
            try:
                prediction = prediction_service.predict_single(mileage, verbose=verbose)

                if verbose:
                    click.echo(f"Predicted price: ${prediction:,.2f}")
                else:
                    # Simple output format
                    click.echo(f"{prediction:.2f}")

            except InvalidArgException as e:
                click.echo(f"Input error: {e}", err=True)
                sys.exit(1)

    except InvalidCsvException as e:
        click.echo(f"Model error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@click.command()
@click.argument("model_file", type=str)
def info(model_file: str) -> None:
    """
    Display information about a trained model.

    MODEL_FILE: Path to trained model file
    """
    try:
        prediction_service = PredictionService(model_file)
        prediction_service.load_model()

        model_info = prediction_service.get_model_info()

        click.echo("=" * 50)
        click.echo("LINEAR REGRESSION MODEL INFORMATION")
        click.echo("=" * 50)
        click.echo(f"Model file: {model_info['model_file']}")
        click.echo(f"Trained: {'Yes' if model_info['is_trained'] else 'No'}")
        click.echo()
        click.echo("PARAMETERS:")
        click.echo(f"  θ₀ (intercept): {model_info['theta0']:.6f}")
        click.echo(f"  θ₁ (slope): {model_info['theta1']:.6f}")
        click.echo()

        if model_info["scaler"]:
            scaler = model_info["scaler"]
            click.echo("TRAINING DATA RANGE:")
            click.echo(
                f"  Mileage: {scaler['km_min']:,.0f} - {scaler['km_max']:,.0f} km"
            )
            click.echo(
                f"  Price: ${scaler['price_min']:,.2f} - ${scaler['price_max']:,.2f}"
            )
            click.echo()

        hyperparams = model_info["hyperparameters"]
        click.echo("HYPERPARAMETERS:")
        click.echo(f"  Learning rate: {hyperparams['learning_rate']}")
        click.echo(f"  Max epochs: {hyperparams['max_epochs']}")
        click.echo(f"  Tolerance: {hyperparams['tolerance']}")
        click.echo()

        # Linear equation
        theta0 = model_info["theta0"]
        theta1 = model_info["theta1"]
        sign = "+" if theta0 >= 0 else "-"
        click.echo("LINEAR EQUATION:")
        click.echo(f"  price = {theta1:.6f} * mileage {sign} {abs(theta0):.6f}")
        click.echo("=" * 50)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Create a group for multiple commands
@click.group()
def cli() -> None:
    """Linear regression prediction tools."""
    pass


cli.add_command(main, name="predict")
cli.add_command(info, name="info")


if __name__ == "__main__":
    main()
