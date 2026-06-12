from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model.predict import predict_output,model,MODEL_VERSION
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse

app = FastAPI()

# Third Improvement
#-----------
# Human Readable
@app.get('/')
def home():
    return {'message': 'Insurance Premium Prediction API'}

# machine readable--> we have to include this because it is needed for cloud deployment 
# when we deploy this on aws, aws requires these info 
@app.get('/health')
def health_check():
    return {
        'status':'ok',
        # Fourth Improvement
        #----------
        'version':MODEL_VERSION,
        'model_loaded':model is not None # is model is loaded above without any error it return true
        #----------
    }
#-----------


@app.post('/predict',response_model=PredictionResponse)
def predict_premium(data: UserInput):

    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    # sixth improvement
    # tryCatch is used because we are using predict_output function which is in external another file which can cause undermine error 
    try:
        prediction = predict_output(user_input)

        return JSONResponse(status_code=200, content=prediction)
    
    except Exception as e:

        return JSONResponse(status_code=500, content={"error": str(e)})




