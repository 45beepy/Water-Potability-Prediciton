import joblib
from flask import Flask, request, jsonify, render_template
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Flask Setup for Model Deployment
app = Flask(__name__)

# Step 2: Load the saved model, transformer, and training data
model = joblib.load('best_model.pkl')
transformer = joblib.load('power_transformer.pkl')
X_train_transformed = joblib.load('X_train_transformed.pkl')

# Step 3: LIME Explainer Setup
explainer = LimeTabularExplainer(
    training_data=X_train_transformed,  # Use the transformed training data
    feature_names=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
    class_names=['Not Potable', 'Potable'],
    discretize_continuous=True
)

# Step 4: LIME Explanation Function
def get_lime_explanation(input_data):
    input_transformed = transformer.transform([input_data])  # Transform input using PowerTransformer
    exp = explainer.explain_instance(input_transformed[0], model.predict_proba, num_features=5)
    return exp.as_html()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Step 5: Get input data from form
    input_data = [float(x) for x in request.form.values()]
    
    # Step 6: Transform the input using PowerTransformer
    input_transformed = transformer.transform([input_data])
    
    # Step 7: Make prediction
    prediction = model.predict(input_transformed)[0]
    
    # Step 8: Get LIME explanation
    explanation = get_lime_explanation(input_data)
    
    # Step 9: Map prediction to output label
    output = 'Potable' if prediction == 1 else 'Not Potable'
    
    return render_template('index.html', prediction_text=f'Water is predicted as {output}', explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)
