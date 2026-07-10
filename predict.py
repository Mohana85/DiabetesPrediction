import numpy as np
import pickle

model=pickle.load(open("diabetes_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))


# User inputs
pregnancies = int(input("Enter Pregnancies: "))
glucose = int(input("Enter Glucose: "))
blood_pressure = int(input("Enter Blood Pressure: "))
skin_thickness = int(input("Enter Skin Thickness: "))
insulin = int(input("Enter Insulin: "))
bmi = float(input("Enter BMI: "))
dpf = float(input("Enter Diabetes Pedigree Function: "))
age = int(input("Enter Age: "))

# Create input array
patient = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]])

patient = scaler.transform(patient)

#Prediction
prediction = model.predict(patient)
probability = model.predict_proba(patient)

if prediction[0] == 1:
    print("\nPrediction: Diabetic")
else:
    print("\nPrediction: Not Diabetic")

print(f"Probability of Diabetes: {probability[0][1] * 100:.2f}%")