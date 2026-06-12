import pickle
import pandas as pd

# import the ml model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)


# model version
# we are setting it manually but in real case we get this version from MLfow sotfware
MODEL_VERSION='1.0.0'

# Get class labels from model (important for matching probabilities to class names)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):

    # seventh improvement is to show confidence score and all probs

    input_df=pd.DataFrame([user_input])
    
    # predict the class
    predicted_class=model.predict(input_df)[0]

    # Get probabilities for all classes
    probabilities = model.predict_proba(input_df)[0]
    confidence= max(probabilities)

    # Create mapping: {class_name: probability}
    class_probs=dict(zip(class_labels,map(lambda p: round(p,4), probabilities)))

    return {

        'predicted_category':predicted_class,
        'confidence':round(confidence,4),
        'class_probabilities':class_probs

    }
