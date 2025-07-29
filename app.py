import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# --- 0. Feature Descriptions ---
# Dictionary mapping feature codes to their meanings for a user-friendly UI
feature_meanings = {
    'cap-shape': {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'},
    'cap-surface': {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'},
    'cap-color': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'bruises': {'t': 'bruises', 'f': 'no'},
    'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'},
    'gill-attachment': {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'},
    'gill-spacing': {'c': 'close', 'w': 'crowded', 'd': 'distant'},
    'gill-size': {'b': 'broad', 'n': 'narrow'},
    'gill-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
    'stalk-root': {'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', '?': 'missing'},
    'stalk-surface-above-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
    'stalk-surface-below-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
    'stalk-color-above-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'stalk-color-below-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
    'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
    'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
    'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'},
    'spore-print-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
    'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
    'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}
}


# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_and_prepare_data():
    """
    Loads the mushroom dataset, preprocesses it, and trains the model.
    Returns the trained model, encoders, and feature options.
    """
    # Load the dataset
    dataset_link = "https://raw.githubusercontent.com/massudavide/Mushroom-Dataset/refs/heads/master/mushroom_data_all.csv"
    df = pd.read_csv(dataset_link)

    # --- Preprocessing Steps from your notebook ---
    # Handle '?' in 'stalk-root'
    stalk_root_mode = df['stalk-root'].mode()[0]
    df['stalk-root'] = df['stalk-root'].replace('?', stalk_root_mode)

    # Drop 'veil-type'
    df.drop('veil-type', axis=1, inplace=True)

    # --- Prepare for Encoding ---
    # Separate features (X) and target (y)
    X = df.drop('class_edible', axis=1)
    y = df['class_edible']

    # Store the original categorical options for the UI
    feature_options = {col: df[col].unique().tolist() for col in X.columns}

    # Encode all features and the target variable
    encoders = {col: LabelEncoder() for col in df.columns}

    df_encoded = pd.DataFrame()
    for col in df.columns:
        df_encoded[col] = encoders[col].fit_transform(df[col])


    X_encoded = df_encoded.drop('class_edible', axis=1)
    y_encoded = df_encoded['class_edible']

    # --- 2. Model Training ---
    # We use the Decision Tree as it had 100% accuracy
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_encoded, y_encoded)

    return model, encoders, feature_options

# Load the trained model, encoders, and options
model, encoders, feature_options = load_and_prepare_data()

# --- 3. Streamlit Frontend ---
st.set_page_config(page_title="Mushroom Edibility Predictor", layout="wide")
st.title("🍄 Mushroom Edibility Predictor")
st.write("Select the features of a mushroom to predict whether it is edible or poisonous. This tool uses a Decision Tree model with 100% accuracy on the original dataset.")

# Create a sidebar for user inputs
st.sidebar.header("Select Mushroom Features")

user_inputs = {}
# Create a selectbox for each feature
for feature, options in feature_options.items():
    # Create a more user-friendly label for the widget
    label = feature.replace('-', ' ').title()

    def format_option(option):
        """Displays 'code: meaning' in the selectbox, e.g., 'f: foul'."""
        # Get the dictionary of meanings for the current feature
        meanings_dict = feature_meanings.get(feature, {})
        # Get the specific meaning for the option; default to the option code itself
        meaning = meanings_dict.get(option, option)
        # Return the formatted string if a meaning was found, otherwise just the code
        return f"{option}: {meaning}" if meaning != option else option

    user_inputs[feature] = st.sidebar.selectbox(
        label,
        options,
        format_func=format_option # This shows descriptive names in the UI
    )


# --- 4. Prediction Logic ---
# Create a button to trigger the prediction
if st.sidebar.button("Predict Edibility", use_container_width=True):
    # Create a DataFrame from the user's inputs
    input_df = pd.DataFrame([user_inputs])

    st.subheader("User Selections:")
    st.dataframe(input_df)

    # Encode the user's input using the same encoders
    input_encoded = pd.DataFrame()
    try:
        for feature in input_df.columns:
            # Get the correct encoder for the feature
            encoder = encoders[feature]
            # Transform the user's input
            # We need to reshape for a single value
            input_encoded[feature] = encoder.transform(input_df[feature].values)

        # Make a prediction
        prediction_encoded = model.predict(input_encoded)

        # Decode the prediction back to a human-readable label
        prediction_label = encoders['class_edible'].inverse_transform(prediction_encoded)[0]

        st.subheader("Prediction Result:")
        if prediction_label == 'e':
            st.success("✅ The mushroom is likely **Edible**.")
            st.balloons()
        else:
            st.error("☠️ The mushroom is likely **Poisonous**.")
            st.warning("Warning: Do not consume wild mushrooms based solely on this prediction. Always consult an expert.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("This might happen if a selected feature value was not seen during model training.")