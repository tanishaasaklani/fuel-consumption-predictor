import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Fuel Consumption Predictor",
    page_icon="🚗",
    layout="centered"
)

df = pd.read_csv(r"D:\dataset\FuelConsumption.csv")

X = df[['ENGINESIZE', 'CYLINDERS']]
y = df[['FUELCONSUMPTION_COMB']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

#UI
st.title("🚗 Fuel Consumption Predictor")
st.caption("Predict fuel consumption using Machine Learning (Polynomial Regression)")

st.markdown("---")

# Inputs
engine_size = st.number_input(
    "🔧 Engine Size (Litres)",
    min_value=0.5,
    max_value=8.0,
    value=2.0,
    step=0.1
)

cylinders = st.selectbox(
    "🔩 Number of Cylinders",
    sorted(df['CYLINDERS'].unique())
)

st.markdown("")

# Validation warning
if engine_size < 1.0 and cylinders >= 6:
    st.warning("⚠️ This engine–cylinder combination is uncommon in real vehicles.")

# Predict button
if st.button("🔍 Predict Fuel Consumption"):

    user_input = pd.DataFrame(
        [[engine_size, cylinders]],
        columns=['ENGINESIZE', 'CYLINDERS']
    )

    user_input_poly = poly.transform(user_input)
    prediction = model.predict(user_input_poly)[0][0]

    # Result
    st.success(f"📊 **Predicted Fuel Consumption:** {prediction:.2f} L/100km")

    # Explanation
    st.markdown(
        "This prediction is based on engine size and cylinder count under "
        "**combined driving conditions**."
    )

    st.markdown("---")

    # Model info
    st.subheader("📈 Model Information")
    st.write(f"**Model:** Polynomial Regression (degree = 2)")
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")

    st.markdown("---")

    #VISUALIZATION
    st.subheader("📉 Fuel Consumption Trend")

    # Fix cylinders for visualization
    fixed_cyl = cylinders
    engine_range = np.linspace(
        df['ENGINESIZE'].min(),
        df['ENGINESIZE'].max(),
        100
    )

    X_vis = pd.DataFrame({
        'ENGINESIZE': engine_range,
        'CYLINDERS': fixed_cyl
    })

    X_vis_poly = poly.transform(X_vis)
    y_vis_pred = model.predict(X_vis_poly)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['ENGINESIZE'], df['FUELCONSUMPTION_COMB'],
               alpha=0.3, label="Actual data")
    ax.plot(engine_range, y_vis_pred,
            color='orange', label="Polynomial trend")
    ax.scatter(engine_size, prediction,
               color='red', s=80, label="Your prediction")

    ax.set_xlabel("Engine Size (L)")
    ax.set_ylabel("Fuel Consumption (L/100km)")
    ax.legend()

    st.pyplot(fig)

    st.markdown("---")
    st.caption("Built with Python, scikit-learn & Streamlit")