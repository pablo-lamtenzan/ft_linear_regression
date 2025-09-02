"""Training program CLI for linear regression."""

import sys
from pathlib import Path

import click

from ..data.exceptions import InvalidCsvException
from ..services.precision import PrecisionService
from ..services.training import TrainingService


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option(
    "--learning-rate",
    "-lr",
    default=0.01,
    type=float,
    help="Learning rate for gradient descent (default: 0.01)",
)
@click.option(
    "--max-epochs",
    "-e",
    default=1000,
    type=int,
    help="Maximum number of training epochs (default: 1000)",
)
@click.option(
    "--tolerance",
    "-t",
    default=1e-6,
    type=float,
    help="Convergence tolerance for early stopping (default: 1e-6)",
)
@click.option(
    "--model-file",
    "-m",
    default="model.json",
    type=str,
    help="Path to save trained model (default: model.json)",
)
@click.option("--no-save", is_flag=True, help="Do not save the trained model")
@click.option(
    "--precision/--no-precision",
    default=False,
    help="Calculate and display precision metrics",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress training output")
def main(
    data_file: Path,
    learning_rate: float,
    max_epochs: int,
    tolerance: float,
    model_file: str,
    no_save: bool,
    precision: bool,
    quiet: bool,
) -> None:
    """
    Train a linear regression model on car price data.

    DATA_FILE: Path to CSV file containing km,price data

    Example:
        train data.csv --learning-rate 0.01 --max-epochs 1000
    """
    try:
        # Validate parameters
        if learning_rate <= 0:
            click.echo("Error: Learning rate must be positive", err=True)
            sys.exit(1)

        if max_epochs <= 0:
            click.echo("Error: Max epochs must be positive", err=True)
            sys.exit(1)

        if tolerance <= 0:
            click.echo("Error: Tolerance must be positive", err=True)
            sys.exit(1)

        # Initialize training service
        training_service = TrainingService(
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            tolerance=tolerance,
            model_save_path=model_file,
        )

        if not quiet:
            click.echo("Training linear regression model...")
            click.echo(f"Data file: {data_file}")
            click.echo(f"Model will be saved to: {model_file}")
            click.echo()

        # Train the model
        model = training_service.train_model(
            str(data_file), save_model=not no_save, verbose=not quiet
        )

        if not quiet:
            click.echo()
            click.echo("✓ Training completed successfully!")

        # Show precision metrics if requested
        if precision:
            if not quiet:
                click.echo()
                click.echo("Calculating precision metrics...")

            precision_service = PrecisionService()
            try:
                report = precision_service.generate_precision_report(
                    str(data_file), model, verbose=not quiet
                )
                if quiet:
                    click.echo(report)
            except Exception as e:
                click.echo(
                    f"Warning: Could not calculate precision metrics: {e}", err=True
                )

        if not quiet:
            click.echo()
            if not no_save:
                click.echo(f"Model saved to: {model_file}")
            click.echo(
                "Training complete! You can now use the predict program to make predictions."
            )

    except InvalidCsvException as e:
        click.echo(f"Data error: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(f"Error: Data file '{data_file}' not found", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
