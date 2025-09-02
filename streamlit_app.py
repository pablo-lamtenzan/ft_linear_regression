#!/usr/bin/env python3
"""
Clean Linear Regression Dashboard

A Streamlit web application that properly uses our tested services
for training and visualizing linear regression models.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.linear_regression.services.precision import PrecisionService

# Import our tested services
from src.linear_regression.services.training import TrainingService


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Linear Regression Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚗 Linear Regression: Car Price Prediction")
    st.markdown("""
    **Interactive dashboard using our tested linear regression services**
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Home",
            "🚀 Train Model",
            "🔮 Make Predictions",
            "📈 Visualizations",
            "📋 Model Analysis",
        ],
    )

    # Initialize session state
    init_session_state()

    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "🚀 Train Model":
        show_training_page()
    elif page == "🔮 Make Predictions":
        show_prediction_page()
    elif page == "📈 Visualizations":
        show_visualization_page()
    elif page == "📋 Model Analysis":
        show_analysis_page()


def init_session_state():
    """Initialize session state variables."""
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "training_data" not in st.session_state:
        st.session_state.training_data = None
    if "training_history" not in st.session_state:
        st.session_state.training_history = []


def show_home_page():
    """Display the home page with project information."""
    st.header("Welcome to the Linear Regression Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 What This App Does")
        st.markdown("""
        - **Train** models
        - **Predict** prices
        - **Analyze** performance
        - **Visualize** everything
        """)

        st.subheader("🧮 Mathematical Foundation")
        st.markdown("""
        **Linear Model:**
        ```
        price = θ₀ + θ₁ × mileage
        ```

        **Where:**
        - `θ₀` (theta0) = intercept (base price)
        - `θ₁` (theta1) = slope (price change per km)
        - `mileage` = car's mileage in kilometers
        """)

    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Upload Data**: Go to "Train with Animation"
        2. **Watch Learning**: See gradient descent in action!
        3. **Make Predictions**: Test your trained model
        4. **Analyze Results**: Explore metrics and visualizations
        """)

        st.subheader("📁 Sample Data Format")
        sample_data = pd.DataFrame(
            {
                "km": [100000, 150000, 200000, 250000, 300000],
                "price": [8000, 7000, 6000, 5000, 4000],
            }
        )
        st.dataframe(sample_data, width="stretch")

        # Download sample data
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Data",
            data=csv,
            file_name="sample_car_data.csv",
            mime="text/csv",
        )


def show_training_page():
    """Display the training page."""
    st.header("🚀 Train Model")

    st.markdown("""
    **Train your linear regression model!** Upload your data and configure the training parameters
    to create a model that predicts car prices based on mileage.
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload training data (CSV format)",
        type=["csv"],
        help="CSV file should have 'km' and 'price' columns",
    )

    if uploaded_file is not None:
        try:
            # Display data preview
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Training Data")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df, width="stretch")
            with col2:
                st.metric("Data Points", len(df))
                st.metric(
                    "Mileage Range", f"{df['km'].min():,.0f} - {df['km'].max():,.0f}"
                )
                st.metric(
                    "Price Range",
                    f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}",
                )

            # Training parameters
            st.subheader("⚙️ Training Parameters")
            col1, col2, col3 = st.columns(3)

            with col1:
                learning_rate = st.slider(
                    "Learning Rate", 0.001, 0.1, 0.01, 0.001, format="%.3f"
                )
            with col2:
                max_epochs = st.slider("Max Epochs", 100, 5000, 1000, 100)
            with col3:
                tolerance = st.selectbox(
                    "Tolerance",
                    [1e-6, 1e-5, 1e-4, 1e-3],
                    index=0,
                    format_func=lambda x: f"{x:.0e}",
                )

            # Train button
            if st.button("🚀 Train Model", type="primary"):
                train_with_animation(
                    uploaded_file, learning_rate, max_epochs, tolerance
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Upload a CSV file to begin training")


def train_with_animation(
    uploaded_file,
    learning_rate: float,
    max_epochs: int,
    tolerance: float,
):
    """Train model with live animation using our TrainingService."""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".csv", delete=False
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        st.subheader("🎯 Live Training Progress")

        # Create containers for real-time updates
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Regression Line Evolution**")
            plot_container = st.empty()
        with col2:
            st.markdown("**Cost Function Progress**")
            cost_container = st.empty()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Load data for visualization and store in session state
        df = pd.read_csv(tmp_file_path)
        mileages = df["km"].values
        prices = df["price"].values

        # Store training data in session state
        st.session_state.training_data = {
            "mileages": mileages,
            "prices": prices,
            "dataframe": df,
        }

        # Create training service
        training_service = TrainingService(
            learning_rate=learning_rate, max_epochs=max_epochs, tolerance=tolerance
        )

        # Train the model
        status_text.text("🚀 Training model...")
        progress_bar.progress(0.5)

        model = training_service.train_model(
            tmp_file_path, save_model=False, verbose=False
        )

        progress_bar.progress(1.0)
        status_text.text("🎉 Training Complete!")

        # Show final visualization
        x_range = np.linspace(mileages.min(), mileages.max(), 100)
        y_pred = [model.predict(x) for x in x_range]

        # Final regression plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=mileages,
                y=prices,
                mode="markers",
                name="Training Data",
                marker={"size": 10, "color": "blue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                name="Regression Line",
                line={"color": "red", "width": 3},
            )
        )
        fig.update_layout(
            title="Final Trained Model",
            xaxis_title="Mileage (km)",
            yaxis_title="Price ($)",
            height=400,
        )
        plot_container.plotly_chart(fig, width="stretch")

        # Show cost history if available
        metrics = model.get_training_metrics()
        if "cost_history" in metrics:
            cost_fig = go.Figure()
            cost_fig.add_trace(
                go.Scatter(
                    y=metrics["cost_history"],
                    mode="lines+markers",
                    name="Cost Function",
                )
            )
            cost_fig.update_layout(
                title="Cost Function Minimization",
                xaxis_title="Epoch",
                yaxis_title="Cost",
                height=400,
            )
            cost_container.plotly_chart(cost_fig, width="stretch")

        # Store trained model
        st.session_state.trained_model = model
        st.session_state.training_data_path = tmp_file_path

        # Update progress
        progress_bar.progress(1.0)
        status_text.text("🎉 Training Complete!")

        # Show results
        st.success("✅ Model trained successfully!")

        # Get training metrics
        metrics = model.get_training_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if model.theta0 is not None:
                st.metric("θ₀ (Intercept)", f"{model.theta0:.6f}")
            else:
                st.metric("θ₀ (Intercept)", "None")
        with col2:
            if model.theta1 is not None:
                st.metric("θ₁ (Slope)", f"{model.theta1:.6f}")
            else:
                st.metric("θ₁ (Slope)", "None")
        with col3:
            if (
                metrics
                and "final_cost" in metrics
                and metrics["final_cost"] is not None
            ):
                st.metric("Final Cost", f"{metrics['final_cost']:.6f}")
            else:
                st.metric("Final Cost", "N/A")
        with col4:
            if (
                metrics
                and "epochs_trained" in metrics
                and metrics["epochs_trained"] is not None
            ):
                st.metric("Epochs", metrics["epochs_trained"])
            else:
                st.metric("Epochs", "N/A")

        # Show mathematical interpretation
        show_model_interpretation(model)

    except Exception as e:
        st.error(f"Training failed: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def show_model_interpretation(model):
    """Show mathematical interpretation of the trained model."""
    st.subheader("🧮 Mathematical Interpretation")

    # Check if model parameters are valid
    if model.theta0 is None or model.theta1 is None:
        st.error("Model parameters are not properly set. Please retrain the model.")
        return

    # Show the equation
    sign = "+" if model.theta0 >= 0 else "-"
    equation = f"price = {model.theta1:.6f} × mileage {sign} {abs(model.theta0):.6f}"
    st.code(equation, language="text")

    # Explain what each parameter means
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parameter Meanings:**")
        st.markdown(f"""
        - **θ₀ = {model.theta0:.6f}**: Base price (intercept)
          - This is the theoretical price of a car with 0 km
        - **θ₁ = {model.theta1:.6f}**: Price change per kilometer
          - For every 1 km increase, price changes by ${model.theta1:.2f}
        """)

    with col2:
        st.markdown("**Practical Examples:**")
        example_km = [50000, 100000, 150000, 200000]
        for km in example_km:
            try:
                predicted_price = model.predict(km)
                if predicted_price is not None:
                    st.markdown(f"- **{km:,} km**: ${predicted_price:,.2f}")
                else:
                    st.markdown(f"- **{km:,} km**: Error - prediction returned None")
            except Exception as e:
                st.markdown(f"- **{km:,} km**: Error - {e}")


def show_prediction_page():
    """Display the prediction interface."""
    st.header("🔮 Make Predictions")

    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first in the 'Train Model' section.")
        return

    model = st.session_state.trained_model

    st.markdown(
        "**Enter a car's mileage to predict its price using your trained model.**"
    )

    # Simple prediction input
    mileage = st.number_input(
        "Car Mileage (km)",
        min_value=0,
        max_value=500000,
        value=150000,
        step=1000,
        help="Enter the mileage of the car you want to predict the price for",
    )

    if st.button("🔮 Predict Price", type="primary", width="stretch"):
        try:
            prediction = model.predict(mileage)

            # Show result prominently
            st.success(f"## 💰 Predicted Price: ${prediction:,.2f}")

            # Show the mathematical calculation
            st.subheader("📐 How this was calculated:")
            st.code(
                f"""
Mathematical Formula: price = θ₁ × mileage + θ₀

Your Values:
• θ₁ (slope) = {model.theta1:.6f}
• θ₀ (intercept) = {model.theta0:.6f}
• mileage = {mileage:,} km

Calculation:
price = {model.theta1:.6f} × {mileage:,} + {model.theta0:.6f}
price = {model.theta1 * mileage:.2f} + {model.theta0:.2f}
price = ${prediction:.2f}
            """,
                language="text",
            )

            # Explain what the parameters mean
            st.info(f"""
            **What do these parameters mean?**

            • **θ₁ = {model.theta1:.6f}**: For every 1 km increase in mileage, the price changes by ${model.theta1:.2f}
            • **θ₀ = {model.theta0:.6f}**: The theoretical base price of a car with 0 km mileage
            """)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


def show_visualization_page():
    """Display interactive visualizations."""
    st.header("📈 Interactive Visualizations")

    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first.")
        return

    model = st.session_state.trained_model

    # Load training data from session state
    if st.session_state.training_data is not None:
        mileages = st.session_state.training_data["mileages"]
        prices = st.session_state.training_data["prices"]
    else:
        # Use sample data
        mileages = np.array([50000, 100000, 150000, 200000, 250000, 300000])
        prices = np.array([9000, 8000, 7000, 6000, 5000, 4000])

    # Interactive regression line plot
    st.subheader("🎯 Data and Regression Line")

    # Create regression line
    x_range = np.linspace(mileages.min(), mileages.max(), 100)
    y_pred = [model.predict(x) for x in x_range]

    fig = go.Figure()

    # Add data points
    fig.add_trace(
        go.Scatter(
            x=mileages,
            y=prices,
            mode="markers",
            name="Training Data",
            marker={"size": 12, "color": "blue"},
            hovertemplate="Mileage: %{x:,} km<br>Price: $%{y:,}<extra></extra>",
        )
    )

    # Add regression line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            name="Regression Line",
            line={"color": "red", "width": 3},
            hovertemplate="Mileage: %{x:,} km<br>Predicted: $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Car Price vs Mileage with Regression Line",
        xaxis_title="Mileage (km)",
        yaxis_title="Price ($)",
        height=500,
        hovermode="closest",
    )

    st.plotly_chart(fig, width="stretch")

    # Interactive prediction point
    st.subheader("🎚️ Interactive Prediction Visualization")

    col1, col2 = st.columns([3, 1])

    with col2:
        interactive_mileage = st.slider(
            "Select Mileage for Prediction",
            min_value=int(mileages.min()),
            max_value=int(mileages.max()),
            value=int(np.mean(mileages)),
            step=5000,
        )

        interactive_prediction = model.predict(interactive_mileage)
        st.metric("Predicted Price", f"${interactive_prediction:,.2f}")

        # Show calculation
        st.markdown(f"""
        **Calculation:**
        ```
        θ₁ × mileage + θ₀
        {model.theta1:.6f} × {interactive_mileage:,} + {model.theta0:.6f}
        = ${interactive_prediction:.2f}
        ```
        """)

    with col1:
        # Update plot with interactive point
        fig_interactive = go.Figure()

        # Add data points
        fig_interactive.add_trace(
            go.Scatter(
                x=mileages,
                y=prices,
                mode="markers",
                name="Training Data",
                marker={"size": 10, "color": "blue", "opacity": 0.7},
            )
        )

        # Add regression line
        fig_interactive.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode="lines",
                name="Regression Line",
                line={"color": "red", "width": 2},
            )
        )

        # Add interactive prediction point
        fig_interactive.add_trace(
            go.Scatter(
                x=[interactive_mileage],
                y=[interactive_prediction],
                mode="markers",
                name="Your Prediction",
                marker={"size": 15, "color": "green", "symbol": "star"},
                hovertemplate=f"Your Input<br>Mileage: {interactive_mileage:,} km<br>Predicted: ${interactive_prediction:,.2f}<extra></extra>",
            )
        )

        fig_interactive.update_layout(
            title="Interactive Prediction Point",
            xaxis_title="Mileage (km)",
            yaxis_title="Price ($)",
            height=400,
        )

        st.plotly_chart(fig_interactive, width="stretch")


def show_analysis_page():
    """Display model analysis and metrics."""
    st.header("📋 Model Analysis")

    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first.")
        return

    model = st.session_state.trained_model

    # Model Parameters
    st.subheader("🔧 Model Parameters")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("θ₀ (Intercept)", f"{model.theta0:.6f}")
        st.metric("θ₁ (Slope)", f"{model.theta1:.6f}")

    with col2:
        # Get training metrics
        metrics = model.get_training_metrics()
        st.metric("Final Cost", f"{metrics['final_cost']:.6f}")
        st.metric("Epochs Trained", metrics["epochs_trained"])

        if metrics["converged"]:
            st.success("✅ Model Converged")
        else:
            st.warning("⚠️ Model Did Not Converge")

    # Linear Equation
    st.subheader("📐 Linear Equation")
    sign = "+" if model.theta0 >= 0 else "-"
    equation = f"price = {model.theta1:.6f} × mileage {sign} {abs(model.theta0):.6f}"
    st.code(equation, language="text")

    # Model Interpretation
    st.subheader("🧠 Model Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parameter Meanings:**")
        st.markdown(f"""
        - **θ₀ = {model.theta0:.6f}**: Base price (intercept)
          - Theoretical price of a car with 0 km
        - **θ₁ = {model.theta1:.6f}**: Price change per kilometer
          - For every 1 km increase, price changes by ${model.theta1:.2f}
        """)

        if model.theta1 < 0:
            st.success(
                f"✅ Negative slope: Price decreases with mileage (${abs(model.theta1):.2f} per km)"
            )
        else:
            st.warning(
                f"⚠️ Positive slope: Price increases with mileage (${model.theta1:.2f} per km)"
            )

    with col2:
        st.markdown("**Practical Examples:**")
        example_km = [50000, 100000, 150000, 200000, 250000]
        for km in example_km:
            try:
                predicted_price = model.predict(km)
                st.markdown(f"- **{km:,} km**: ${predicted_price:,.2f}")
            except Exception:
                st.markdown(f"- **{km:,} km**: Error in prediction")

    # Performance Analysis using PrecisionService
    st.subheader("� Performance Analysis")

    if st.session_state.training_data is not None:
        try:
            # Create temporary file for PrecisionService
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp_file:
                df = st.session_state.training_data["dataframe"]
                df.to_csv(tmp_file.name, index=False)
                tmp_path = tmp_file.name

            # Use our PrecisionService for analysis
            precision_service = PrecisionService()

            # Calculate comprehensive metrics
            comprehensive_metrics = precision_service.calculate_comprehensive_metrics(
                tmp_path, model
            )

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("R² Score", f"{comprehensive_metrics['r_squared']:.4f}")
            with col2:
                st.metric("MAE", f"${comprehensive_metrics['mae']:.2f}")
            with col3:
                st.metric("RMSE", f"${comprehensive_metrics['rmse']:.2f}")
            with col4:
                st.metric("MAPE", f"{comprehensive_metrics['mape']:.2f}%")

            # Model quality assessment
            r_squared = comprehensive_metrics["r_squared"]
            if r_squared > 0.8:
                st.success(f"🎯 Excellent fit (R² = {r_squared:.3f})")
            elif r_squared > 0.6:
                st.warning(f"👍 Good fit (R² = {r_squared:.3f})")
            else:
                st.error(f"👎 Poor fit (R² = {r_squared:.3f})")

            # Generate precision report
            st.subheader("📋 Detailed Precision Report")

            with st.expander("View Full Precision Report"):
                report = precision_service.generate_precision_report(
                    tmp_path, model, verbose=False
                )
                st.text(report)

            # Clean up temporary file
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Could not calculate precision metrics: {e}")
            st.info("Using basic metrics from training data...")

            # Fallback to basic analysis
            try:
                mileages = st.session_state.training_data["mileages"]
                prices = st.session_state.training_data["prices"]
                predictions = [model.predict(km) for km in mileages]

                # Calculate basic R²
                ss_res = np.sum((prices - predictions) ** 2)
                ss_tot = np.sum((prices - np.mean(prices)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                st.metric("R² Score (Basic)", f"{r_squared:.4f}")

            except Exception as e2:
                st.error(f"Could not perform basic analysis: {e2}")
    else:
        st.info("No training data available for precision analysis.")

    # Training History
    st.subheader("📈 Training History")

    if "cost_history" in metrics and metrics["cost_history"]:
        cost_history = metrics["cost_history"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=cost_history,
                mode="lines+markers",
                name="Cost Function",
                line={"color": "blue", "width": 2},
            )
        )

        fig.update_layout(
            title="Cost Function During Training",
            xaxis_title="Epoch",
            yaxis_title="Cost",
            height=400,
        )

        st.plotly_chart(fig, width="stretch")

        # Show convergence info
        if len(cost_history) > 1:
            initial_cost = cost_history[0]
            final_cost = cost_history[-1]
            improvement = ((initial_cost - final_cost) / initial_cost) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Cost", f"{initial_cost:.6f}")
            with col2:
                st.metric("Final Cost", f"{final_cost:.6f}")
            with col3:
                st.metric("Improvement", f"{improvement:.2f}%")
    else:
        st.info("No cost history available from training.")


if __name__ == "__main__":
    main()
