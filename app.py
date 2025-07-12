from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import pickle
from collections import Counter
import os

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Replace with a strong secret key for session management

# --- Load Models ---
# IMPORTANT: Adjust these paths if your 'models' directory is not directly inside your Flask app directory
# For deployment, consider absolute paths or a more robust path handling
try:
    with open(os.path.join(os.path.dirname(__file__), 'models', 'logistic_model.pkl'), "rb") as f1:
        logistic_model = pickle.load(f1)
    with open(os.path.join(os.path.dirname(__file__), 'models', 'nb_model.pkl'), "rb") as f2:
        nb_model = pickle.load(f2)
    with open(os.path.join(os.path.dirname(__file__), 'models', 'sgd_model.pkl'), "rb") as f3:
        sgd_model = pickle.load(f3)
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Make sure 'models' directory and .pkl files are in the correct path.")
    # Exit or handle gracefully, e.g., run without models or provide a dummy response
    logistic_model = None
    nb_model = None
    sgd_model = None


# --- Load Dataset ---
try:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'skincareproducts.csv'))
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}. Make sure 'skincareproducts.csv' is in the correct path.")
    df = pd.DataFrame() # Empty DataFrame if not found


# --- Valid Options ---
valid_options = {
    "skin_type": ["Oily", "Dry", "Normal", "Combination"],
    "skin_tone": ["Fair", "Medium", "Olive", "Dark"],
    "hair_color": ["Black", "Brown", "Blonde", "Red", "Other"],
    "eye_color": ["Brown", "Blue", "Green", "Hazel", "Other"]
}

# --- Predict Function ---
def predict_category(models, skin_type, skin_tone, hair_color, eye_color):
    # Ensure models are loaded before attempting prediction
    if not all(models):
        return "Unknown Category (models not loaded)"

    input_df = pd.DataFrame({
        "skin_type": [skin_type],
        "skin_tone": [skin_tone],
        "hair_color": [hair_color],
        "eye_color": [eye_color]
    })
    predictions = [model.predict(input_df)[0] for model in models if model is not None]
    if predictions:
        most_common = Counter(predictions).most_common(1)[0][0]
    else:
        most_common = "General Skincare" # Fallback if no predictions are made
    return most_common

# --- Chat Questions ---
def get_question(step):
    questions = {
        1: "🌸 Hi there! I'm RoopBot. What’s your **skin type**? (Oily, Dry, Normal, Combination)",
        2: "🌼 Thanks! What’s your **skin tone**? (Fair, Medium, Olive, Dark)",
        3: "💇‍♀️ Awesome! What’s your **hair color**? (Black, Brown, Blonde, Red, Other)",
        4: "👁️ Great! What’s your **eye color**? (Brown, Blue, Green, Hazel, Other)"
    }
    return questions.get(step, "")

@app.route("/")
def home():
    # Initialize session state if not already present
    if "step" not in session:
        session["step"] = 1
        session["skin_type"] = ""
        session["skin_tone"] = ""
        session["hair_color"] = ""
        session["eye_color"] = ""
        session["messages"] = [] # To store chat history displayed on reload

    # Initial bot message
    if not session["messages"]:
        session["messages"].append({"role": "bot", "content": get_question(session["step"])})

    return render_template("index.html", messages=session["messages"])

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    user_text_processed = user_text.strip().capitalize() # Process user input

    response = ""
    current_step = session.get("step", 1)

    session["messages"].append({"role": "user", "content": user_text}) # Add user message to history

    key_map = {1: "skin_type", 2: "skin_tone", 3: "hair_color", 4: "eye_color"}
    current_key = key_map.get(current_step)

    if current_key:
        valid_vals = valid_options[current_key]
        if user_text_processed not in valid_vals:
            response = f"⚠️ Oops! Please choose from: {', '.join(valid_vals)}"
        else:
            session[current_key] = user_text_processed
            session["step"] += 1
            if session["step"] <= 4:
                response = get_question(session["step"])
            else:
                # Final step: Recommend products
                response = "🧴 Let me find the perfect products for your features..."

                models = [logistic_model, nb_model, sgd_model]
                predicted_category = predict_category(
                    models,
                    session["skin_type"],
                    session["skin_tone"],
                    session["hair_color"],
                    session["eye_color"]
                )

                if not df.empty:
                    top_products = (
                        df[df["Category"] == predicted_category]
                        .drop_duplicates(subset=["product_name"])
                        .sort_values(by="Rating_Stars", ascending=False)
                        .head(3)
                    )

                    if not top_products.empty:
                        response += f"<br><br>🌟 I recommend these **{predicted_category}** products for you!<br>"
                        for _, row in top_products.iterrows():
                            response += f"""
<br><b>🛍️ {row['product_name']}</b><br>
⭐ <b>Rating</b>: {row['Rating_Stars']}<br>
🔗 <a href="{row['Product_Url']}" target="_blank">View Product</a><br>
"""
                        response += "<br>💖 Hope you glow even more with these! Type 'restart' to try again."
                    else:
                        response += f"<br><br>Sorry, I couldn't find any **{predicted_category}** products based on your input. Please try again or provide different details."
                else:
                    response += "<br><br>Sorry, I'm having trouble accessing product data right now. Please try again later."
                session["step"] = 5 # Mark as completed, allow restart

    if user_text.lower() == "restart":
        session.clear()
        session["step"] = 1
        session["skin_type"] = ""
        session["skin_tone"] = ""
        session["hair_color"] = ""
        session["eye_color"] = ""
        session["messages"] = []
        response = get_question(session["step"])
        session["messages"].append({"role": "bot", "content": response})


    session["messages"].append({"role": "bot", "content": response}) # Add bot response to history
    return response

if __name__ == "__main__":
    app.run(debug=True) # Set debug=False for production
    