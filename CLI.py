import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data = pd.read_csv('Student Mental health .csv')

# Data Preprocessing
data = data.rename(columns=lambda x: x.strip())
data = data.drop(columns=['Timestamp'])  # Drop timestamp as it's not needed for prediction

# Handle missing values
data = data.dropna()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
features = data.drop(columns=['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?'])
target_depression = data['Do you have Depression?']
target_anxiety = data['Do you have Anxiety?']
target_panic_attack = data['Do you have Panic attack?']

# Split the data
X_train_depression, X_test_depression, y_train_depression, y_test_depression = train_test_split(features, target_depression, test_size=0.2, random_state=42)
X_train_anxiety, X_test_anxiety, y_train_anxiety, y_test_anxiety = train_test_split(features, target_anxiety, test_size=0.2, random_state=42)
X_train_panic, X_test_panic, y_train_panic, y_test_panic = train_test_split(features, target_panic_attack, test_size=0.2, random_state=42)

# Train the model
model_depression = RandomForestClassifier(random_state=42)
model_anxiety = RandomForestClassifier(random_state=42)
model_panic_attack = RandomForestClassifier(random_state=42)

model_depression.fit(X_train_depression, y_train_depression)
model_anxiety.fit(X_train_anxiety, y_train_anxiety)
model_panic_attack.fit(X_train_panic, y_train_panic)

# Predict and evaluate
y_pred_depression = model_depression.predict(X_test_depression)
y_pred_anxiety = model_anxiety.predict(X_test_anxiety)
y_pred_panic_attack = model_panic_attack.predict(X_test_panic)

accuracy_depression = accuracy_score(y_test_depression, y_pred_depression)
accuracy_anxiety = accuracy_score(y_test_anxiety, y_pred_anxiety)
accuracy_panic_attack = accuracy_score(y_test_panic, y_pred_panic_attack)

print(f'Accuracy for Depression Prediction: {accuracy_depression}')
print(f'Accuracy for Anxiety Prediction: {accuracy_anxiety}')
print(f'Accuracy for Panic Attack Prediction: {accuracy_panic_attack}')

def collect_user_input():
    input_data = {}

    # Gender input
    print("\nChoose your gender:")
    print("1. Female")
    print("2. Male")
    gender_choice = input("Enter your choice (1 or 2): ").strip()
    if gender_choice == '1':
        input_data['Choose your gender'] = 'Female'
    elif gender_choice == '2':
        input_data['Choose your gender'] = 'Male'
    else:
        print("Invalid choice.")
        return None  # Return None to handle invalid inputs

    # Age input
    input_data['Age'] = int(input("\nWhat is your age? ").strip())

    # Course of study input
    print("\nChoose your course of study:")
    print("1. Accounting")
    print("2. Applied Liberal Arts")
    print("3. Banking studies")
    print("4. Biomedical science")
    print("5. Biotechnology")
    print("6. Business administration")
    print("7. Communication")
    print("8. Computer science (bachelor)")
    print("9. Economics")
    print("10. Economics & mangement")
    print("11. Engineering")
    print("12. Health science (master)")
    print("13. Human resources")
    print("14. Human sciences")
    print("15. Human sciences (english & literature) (bachelor)")
    print("16. Information technology (bachelor)")
    print("17. Islamic education")
    print("18. Law")
    print("19. Marine science")
    print("20. Market access learning compendium")
    print("21. Mathemathics")
    print("22. Nursing (Diploma)")
    print("23. Pharmacy")
    print("24. Psychology")
    print("25. Radiography")
    print("26. Teaching (Diploma)")
    print("27. Transcultural studies")

    course_choice = input("Enter your choice (1 to 27): ").strip()
    if course_choice == '1':
        input_data['What is your course?'] = 'Accounting'
    elif course_choice == '2':
        input_data['What is your course?'] = 'Applied Liberal Arts'
    elif course_choice == '3':
        input_data['What is your course?'] = 'Banking studies'
    elif course_choice == '4':
        input_data['What is your course?'] = 'Biomedical science'
    elif course_choice == '5':
        input_data['What is your course?'] = 'Biotechnology'
    elif course_choice == '6':
        input_data['What is your course?'] = 'Business administration'
    elif course_choice == '7':
        input_data['What is your course?'] = 'Communication'
    elif course_choice == '8':
        input_data['What is your course?'] = 'Computer science (bachelor)'
    elif course_choice == '9':
        input_data['What is your course?'] = 'Economics'
    elif course_choice == '10':
        input_data['What is your course?'] = 'Economics & mangement'
    elif course_choice == '11':
        input_data['What is your course?'] = 'Engineering'
    elif course_choice == '12':
        input_data['What is your course?'] = 'Health science (master)'
    elif course_choice == '13':
        input_data['What is your course?'] = 'Human resources'
    elif course_choice == '14':
        input_data['What is your course?'] = 'Human sciences'
    elif course_choice == '15':
        input_data['What is your course?'] = 'Human sciences (english & literature) (bachelor)'
    elif course_choice == '16':
        input_data['What is your course?'] = 'Information technology (bachelor)'
    elif course_choice == '17':
        input_data['What is your course?'] = 'Islamic education'
    elif course_choice == '18':
        input_data['What is your course?'] = 'Law'
    elif course_choice == '19':
        input_data['What is your course?'] = 'Marine science'
    elif course_choice == '20':
        input_data['What is your course?'] = 'Market access learning compendium'
    elif course_choice == '21':
        input_data['What is your course?'] = 'Mathemathics'
    elif course_choice == '22':
        input_data['What is your course?'] = 'Nursing (Diploma)'
    elif course_choice == '23':
        input_data['What is your course?'] = 'Pharmacy'
    elif course_choice == '24':
        input_data['What is your course?'] = 'Psychology'
    elif course_choice == '25':
        input_data['What is your course?'] = 'Radiography'
    elif course_choice == '26':
        input_data['What is your course?'] = 'Teaching (Diploma)'
    elif course_choice == '27':
        input_data['What is your course?'] = 'Transcultural studies'
    else:
        print("Invalid choice.")
        return None

    # Year of study input
    print("\nChoose your current year of study:")
    print("1. Year 1")
    print("2. Year 2")
    print("3. Year 3")
    print("4. Year 4")

    year_choice = input("Enter your choice (1 to 4): ").strip()
    if year_choice == '1':
        input_data['Your current year of Study'] = 'Year 1'
    elif year_choice == '2':
        input_data['Your current year of Study'] = 'Year 2'
    elif year_choice == '3':
        input_data['Your current year of Study'] = 'Year 3'
    elif year_choice == '4':
        input_data['Your current year of Study'] = 'Year 4'
    else:
        print("Invalid choice.")
        return None

    # CGPA input
    while True:
        cgpa_input = input("What is your CGPA (e.g., 3.00 - 3.49)? ").strip()
        try:
            cgpa = float(cgpa_input)
            if 0.0 <= cgpa <= 4.0:
                input_data['What is your CGPA?'] = cgpa_input
                break
            else:
                print("Invalid CGPA. Please enter a CGPA between 0.0 and 4.0.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for your CGPA.")

    # Marital status input
    print("\nChoose your marital status:")
    print("1. No")
    print("2. Yes")

    marital_choice = input("Enter your choice (1 or 2): ").strip()
    if marital_choice == '1':
        input_data['Marital status'] = 'No'
    elif marital_choice == '2':
        input_data['Marital status'] = 'Yes'
    else:
        print("Invalid choice.")
        return None

    # Specialist treatment input
    print("\nDid you seek any specialist for a treatment?")
    print("1. Yes")
    print("2. No")

    treatment_choice = input("Enter your choice (1 or 2): ").strip()
    if treatment_choice == '1':
        input_data['Did you seek any specialist for a treatment?'] = 'Yes'
    elif treatment_choice == '2':
        input_data['Did you seek any specialist for a treatment?'] = 'No'
    else:
        print("Invalid choice.")
        return None

    return input_data

def handle_unseen_labels(column, value):
    if value not in label_encoders[column].classes_:
        label_encoders[column].classes_ = np.append(label_encoders[column].classes_, value)
    return label_encoders[column].transform([value])[0]

def provide_recommendations(probability, condition):
    likelihood = probability[1]  # Probability of having the condition
    if likelihood > 0.5:  # If the model predicts the presence of a mental health issue
        return f"\nBased on your responses, it seems you might be experiencing some {condition} with a likelihood of {likelihood:.2f}. Here are some recommendations to help alleviate it:\n"\
               "1. Talk to a mental health professional.\n"\
               "2. Engage in regular physical activity.\n"\
               "3. Maintain a healthy sleep routine.\n"\
               "4. Practice mindfulness and relaxation techniques.\n"\
               "5. Stay connected with friends and family."
    else:
        return f"\nBased on your responses, it seems you might not be experiencing significant {condition} with a likelihood of {likelihood:.2f}. "\
               "However, it's important to maintain good mental health practices:\n"\
               "1. Keep a balanced diet and exercise regularly.\n"\
               "2. Ensure you get enough sleep.\n"\
               "3. Take breaks and manage stress effectively.\n"\
               "4. Stay connected with your support network.\n"\
               "5. Don't hesitate to seek help if you feel overwhelmed."

def predict_and_recommend():
    input_data = collect_user_input()

    if input_data is None:
        print("Invalid input, unable to provide recommendations.")
        return

    # Encode the input data
    for column in input_data:
        if column in label_encoders:
            input_data[column] = handle_unseen_labels(column, input_data[column])
    input_df = pd.DataFrame([input_data])

    # Predict probabilities
    depression_prob = model_depression.predict_proba(input_df)[0]
    anxiety_prob = model_anxiety.predict_proba(input_df)[0]
    panic_prob = model_panic_attack.predict_proba(input_df)[0]

    # Provide recommendations
    recommendations = {
        'Depression': provide_recommendations(depression_prob, "depression"),
        'Anxiety': provide_recommendations(anxiety_prob, "anxiety"),
        'Panic Attack': provide_recommendations(panic_prob, "panic attacks")
    }

    # Print probabilities
    print(f"\nProbability of having Depression: {depression_prob[1]:.2f}")
    print(f"Probability of having Anxiety: {anxiety_prob[1]:.2f}")
    print(f"Probability of having Panic Attack: {panic_prob[1]:.2f}")

    return recommendations

# Get predictions and recommendations
recommendations = predict_and_recommend()
if recommendations:
    for condition, recommendation in recommendations.items():
        print(f"\n{condition} Recommendation:\n{recommendation}\n")
