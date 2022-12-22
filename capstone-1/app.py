import pickle as p
# import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction import DictVectorizer
from waitress import serve

app = Flask('Stroke-prediction')


def data_prep(payload):
    stroke_data = {}

    # heart disease data
    if payload["Heart_disease"].upper() == "NO":
        stroke_data["heart_disease"] = 0
    elif payload["Heart_disease"].upper() == "YES":
        stroke_data["heart_disease"] = 1
    else:  # default to No
        stroke_data["heart_disease"] = 0

    # Glucose level data
    if payload["Avg_glucose_level"] > 0:
        stroke_data["avg_glucose_level"] = payload["Avg_glucose_level"]
    else:
        stroke_data["avg_glucose_level"] = 100

    # Hypertension data
    if payload["Hypertension"].upper() == "NO":
        stroke_data["hypertension"] = 0
    elif payload["Hypertension"].upper() == "YES":
        stroke_data["hypertension"] = 1
    else:  # default to No
        stroke_data["hypertension"] = 0

    stroke_data["age"] = payload["Age"]

    # BMI data
    if payload["Bmi"] > 0:
        stroke_data["bmi"] = payload["Bmi"]
    else:
        stroke_data["bmi"] = 22

    # gender data
    if payload["Gender"].upper() == "MALE":
        stroke_data["gender_Male"] = 1
        stroke_data["gender_Other"] = 0
    elif payload["Gender"].upper() == "FEMALE":
        stroke_data["gender_Male"] = 0
        stroke_data["gender_Other"] = 0
    else:
        stroke_data["gender_Male"] = 0
        stroke_data["gender_Other"] = 1

    # marriage data
    if payload["Ever_married"].upper() == "NO":
        stroke_data["ever_married_Yes"] = 0
    elif payload["Ever_married"].upper() == "YES":
        stroke_data["ever_married_Yes"] = 1
    else:  # default to No
        stroke_data["ever_married_Yes"] = 0

    # work type data
    if payload["Work_type"].upper() == "SELF-EMPLOYED":
        stroke_data["work_type_Never_worked"] = 0
        stroke_data["work_type_Private"] = 0
        stroke_data["work_type_Self-employed"] = 1
        stroke_data["work_type_children"] = 0
    elif payload["Work_type"].upper() == "NEVER WORKED":
        stroke_data["work_type_Never_worked"] = 1
        stroke_data["work_type_Private"] = 0
        stroke_data["work_type_Self-employed"] = 0
        stroke_data["work_type_children"] = 0

    elif payload["Work_type"].upper() == "PRIVATE":
        stroke_data["work_type_Never_worked"] = 0
        stroke_data["work_type_Private"] = 1
        stroke_data["work_type_Self-employed"] = 0
        stroke_data["work_type_children"] = 0
    elif payload["Work_type"].upper() == "CHILDREN":
        stroke_data["work_type_Never_worked"] = 0
        stroke_data["work_type_Private"] = 0
        stroke_data["work_type_Self-employed"] = 0
        stroke_data["work_type_children"] = 1
    else:
        stroke_data["work_type_Never_worked"] = 0
        stroke_data["work_type_Private"] = 0
        stroke_data["work_type_Self-employed"] = 0
        stroke_data["work_type_children"] = 0

    # residence data
    if payload["Urban_residence"].upper() == "NO":
        stroke_data["Residence_type_Urban"] = 0
    elif payload["Urban_residence"].upper() == "YES":
        stroke_data["Residence_type_Urban"] = 1
    else:  # default to Yes
        stroke_data["Residence_type_Urban"] = 1

    if payload["Smoking_status"].upper() == "FORMERLY SMOKED":
        stroke_data["smoking_status_formerly smoked"] = 1
        stroke_data["smoking_status_never smoked"] = 0
        stroke_data["smoking_status_smokes"] = 0
    elif payload["Smoking_status"].upper() == "NEVER SMOKED":
        stroke_data["smoking_status_formerly smoked"] = 0
        stroke_data["smoking_status_never smoked"] = 1
        stroke_data["smoking_status_smokes"] = 0
    else:
        stroke_data["smoking_status_formerly smoked"] = 0
        stroke_data["smoking_status_never smoked"] = 0
        stroke_data["smoking_status_smokes"] = 1

    return stroke_data


if __name__ == '__main__':
    model_file = 'models/stroke_predict_rfc.pkl'
    model = p.load(open(model_file, 'rb'))
    # app.run(debug=True, host='0.0.0.0', port=9696
    serve(app, port=9696, host="0.0.0.0")


@app.route('/api/stroke-predict', methods=['POST'])
def predict():
    answer = ""

    payload = request.get_json()
    print(payload)

    stroke_data = data_prep(payload)
    print("stroke data after transform: ", stroke_data)

    v = DictVectorizer(sparse=False)

    # stroke_df = pd.DataFrame([stroke_data])
    stroke_df = v.fit_transform(stroke_data)
    # prediction = np.array2string(model.predict(stroke_df))
    prediction = model.predict(stroke_df)

    if prediction[0] == 0:
        answer = "Congrats! You don't have a risk of stroke for now"
    elif prediction[0] == 1:
        answer = "You have a risk of stroke! Stay a healthy life"
    else:
        answer = "OK"
    # return prediction
    return jsonify(answer)
