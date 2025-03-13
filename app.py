from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# gender: ['female' 'male']
# race_ethnicity: ['group B' 'group C' 'group A' 'group D' 'group E']
# parental_level_of_education: ["bachelor's degree" 'some college' "master's degree" "associate's degree"
#  'high school' 'some high school']
# lunch: ['standard' 'free/reduced']
# test_preparation_course: ['none' 'completed']


# manual inputs
gender = "male"
ethnicity = "group B"
parental_education = "bachelor's degree"
lunch = "standard"
test_preparation_course = "completed"
reading_score = 88
writing_score = 78


def predict_datapoint():
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=float(reading_score),
        writing_score=float(writing_score),
    )

    pred_df = data.get_data_as_dataframe()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("after Prediction")
    return results


if __name__ == "__main__":
    results = predict_datapoint()
    print(f"Predicted Math Score: {results[0]}")
