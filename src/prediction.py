from preprocessing import grayscale_luminosity
from features import extract_features

def predict_image(model, image_path):
    gray = grayscale_luminosity(image_path)

    features = extract_features(image_path, gray)

    predictions = model.predict(features)

    predicted_mask = predictions.reshape(gray.shape)

    return predicted_mask