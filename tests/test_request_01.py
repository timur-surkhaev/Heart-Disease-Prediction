# python3.11 test_request_01.py

import requests

url = 'http://127.0.0.1:8000/predict'

patient = {
           "BMI":24.3,
           "Smoking":"No",
           "AlcoholDrinking":"No",
           "Stroke":"No",
           "PhysicalHealth":0.0,
           "MentalHealth":15.0,
           "DiffWalking":"No",
           "Sex":"Female",
           "AgeCategory":"40-44",
           "Race":"White",
           "Diabetic":"No",
           "PhysicalActivity":"Yes",
           "GenHealth":"Excellent",
           "SleepTime":7.0,
           "Asthma":"No",
           "KidneyDisease":"No",
           "SkinCancer":"No"
            }

response = requests.post(url, json=patient).json()

print(response)


