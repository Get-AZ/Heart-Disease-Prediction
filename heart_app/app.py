# app.py
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="centered",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_FILES = {
    "XGBoost": "xgboost_model.pkl",
    "Random Forest": "randomforest_model.pkl",
    "Decision Tree": "decisiontree_model.pkl",
}

# Raw feature names and order used during training.
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


@st.cache_resource
def load_artifacts(model_name):
    """Load artifacts copied from the same training run."""
    required_files = {
        "preprocessor": MODEL_DIR / "preprocessor.pkl",
        "selector": MODEL_DIR / "selector.pkl",
        "model": MODEL_DIR / MODEL_FILES[model_name],
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    artifacts = {}

    # Load only your own trusted model files.
    for name, path in required_files.items():
        with open(path, "rb") as file:
            artifacts[name] = pickle.load(file)

    return (
        artifacts["preprocessor"],
        artifacts["selector"],
        artifacts["model"],
    )


def prepare_input(raw_data):
    """Apply the same zero-value handling used in training."""
    prepared = raw_data[FEATURE_COLUMNS].copy()

    prepared[["RestingBP", "Cholesterol"]] = (
        prepared[["RestingBP", "Cholesterol"]]
        .replace(0, np.nan)
    )

    return prepared


st.title("❤️ Heart Disease Classification")
st.write(
    "Enter the dataset features to explore predictions "
    "from your trained classification models."
)

st.warning(
    "Educational project only. Predictions are not a medical "
    "diagnosis or a substitute for professional medical care."
)

with st.sidebar:
    st.header("Model settings")

    model_name = st.selectbox(
        "Choose a trained model",
        list(MODEL_FILES),
    )

    st.caption(
        "Each model uses the saved preprocessor and feature selector."
    )

    st.info(
        "Copy all model artifacts from the same training run. "
        "Mixing files from different runs can produce incorrect results."
    )

try:
    preprocessor, selector, model = load_artifacts(model_name)
except Exception as error:
    st.error(f"Could not load model artifacts:\n\n{error}")
    st.stop()


with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=54,
            step=1,
        )

        sex = st.selectbox(
            "Sex",
            ["M", "F"],
            format_func=lambda value: {
                "M": "Male",
                "F": "Female",
            }[value],
        )

        chest_pain = st.selectbox(
            "Chest pain type",
            ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda value: {
                "ASY": "ASY — Asymptomatic",
                "ATA": "ATA — Atypical angina",
                "NAP": "NAP — Non-anginal pain",
                "TA": "TA — Typical angina",
            }[value],
        )

        resting_bp = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=0,
            max_value=300,
            value=130,
            step=1,
            help=(
                "Enter 0 if unrecorded. The saved training "
                "imputer will fill the missing value."
            ),
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=1000,
            value=223,
            step=1,
            help=(
                "Enter 0 if unrecorded. The saved training "
                "imputer will fill the missing value."
            ),
        )

        fasting_bs = st.selectbox(
            "Fasting blood sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda value: {
                0: "No — 0",
                1: "Yes — 1",
            }[value],
        )

    with right:
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            format_func=lambda value: {
                "Normal": "Normal",
                "ST": "ST — ST-T wave abnormality",
                "LVH": "LVH — Left ventricular hypertrophy",
            }[value],
        )

        max_hr = st.number_input(
            "Maximum heart rate achieved (bpm)",
            min_value=1,
            max_value=300,
            value=140,
            step=1,
        )

        exercise_angina = st.selectbox(
            "Exercise-induced angina",
            ["N", "Y"],
            format_func=lambda value: {
                "N": "No",
                "Y": "Yes",
            }[value],
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-10.0,
            max_value=15.0,
            value=1.0,
            step=0.1,
            help="ST depression value recorded in the dataset.",
        )

        st_slope = st.selectbox(
            "ST slope",
            ["Up", "Flat", "Down"],
            format_func=lambda value: {
                "Up": "Up — Upsloping",
                "Flat": "Flat",
                "Down": "Down — Downsloping",
            }[value],
        )

    submitted = st.form_submit_button("Predict")


if submitted:
    raw_input = pd.DataFrame(
        [{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }],
        columns=FEATURE_COLUMNS,
    )

    try:
        prepared_input = prepare_input(raw_input)

        # Never fit preprocessing again during prediction.
        encoded_input = preprocessor.transform(prepared_input)
        selected_input = selector.transform(encoded_input)

        expected_features = getattr(model, "n_features_in_", None)

        if (
            expected_features is not None
            and selected_input.shape[1] != expected_features
        ):
            raise ValueError(
                f"Model expects {expected_features} features, "
                f"but preprocessing produced "
                f"{selected_input.shape[1]}. "
                "Copy matching artifacts from the same training run."
            )

        prediction = int(model.predict(selected_input)[0])

        positive_index = list(model.classes_).index(1)
        probability = float(
            model.predict_proba(selected_input)[0, positive_index]
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

    else:
        st.subheader("Prediction result")

        if prediction == 1:
            st.warning("Predicted dataset class: HeartDisease = 1")
        else:
            st.info("Predicted dataset class: HeartDisease = 0")

        st.metric(
            "Model probability for class 1",
            f"{probability:.1%}",
        )

        st.caption(
            f"Model: {model_name}. This is the model's probability "
            "estimate; it has not been validated as an individual "
            "clinical risk percentage."
        )

        if resting_bp == 0 or cholesterol == 0:
            st.info(
                "Zero blood pressure or cholesterol values were "
                "treated as missing and filled using training-set medians."
            )

        with st.expander("View entered features"):
            st.dataframe(raw_input, hide_index=True)

        result = raw_input.copy()
        result["Model"] = model_name
        result["PredictedClass"] = prediction
        result["Class1Probability"] = probability

        st.download_button(
            "Download prediction as CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="heart_disease_prediction.csv",
            mime="text/csv",
        )