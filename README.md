# Intern Performance Analytics Dashboard

A predictive ML project that evaluates and forecasts intern performance using key metrics such as completion time, feedback rating, and attendance. The project uses an XGBoost regressor and exposes a simple Gradio dashboard where users can input intern details and view predicted performance scores.

## Project Overview

This repository is designed to help teams assess whether an intern is likely to perform well based on observable indicators. It combines a machine learning model with an interactive dashboard so the prediction can be used in a real-world evaluation workflow.

The model predicts a performance score and labels each intern as:

- `🌟 Likely to Excel` when the predicted score is 69 or above
- `⚠️ Likely to Struggle` when the predicted score is below 69

## Why This Project Matters

Intern performance evaluation often depends on multiple factors. This project converts those factors into a data-driven prediction system, making intern assessment more objective and consistent. It is useful for:

- HR and team leads monitoring trainee performance
- Identifying interns needing support early
- Tracking performance trends over time
- Supporting decision-making based on measurable metrics

## Features

- Predictive model for intern performance scoring
- Gradio-based interactive dashboard
- Add new intern records dynamically
- Real-time prediction and classification
- Data table showing all interns and statuses
- Model saved in JSON format for reuse

## Tech Stack

- Python
- Pandas
- XGBoost
- Gradio
- Jupyter Notebook

## Project Structure

```text
intern-performance/
├── app.py                          # Gradio dashboard and prediction logic
├── employee_performance.ipynb      # Model training notebook
├── intern_dataset_realistic (1).csv # Dataset used for model development
├── intern_performance_model.json    # Trained XGBoost model
├── requirements.txt                # Python dependencies
└── README.md                      # Project documentation
```

## Model Details

The trained model uses the following input features:

- `Completion_Time`
- `Feedback_Rating`
- `Attendance`

The target output is:

- `Performance_Score`

The model is trained using XGBoost Regressor and then serialized using `save_model()` in JSON format so it can be loaded directly in the app.

## How It Works

1. A dataset of intern records is loaded and cleaned.
2. Features are separated from the target performance score.
3. The data is split into training and testing sets.
4. An XGBoost regressor is trained.
5. The trained model is saved as `intern_performance_model.json`.
6. The Gradio app loads the model and predicts performance for new intern entries.

## Dashboard Functionality

The application includes:

- A form to enter intern details
- Inputs for:
  - Name
  - Completion time (days)
  - Feedback rating
  - Attendance percentage
- A button to predict and add the intern to the dashboard
- A table showing all interns with calculated scores and labels

## Installation

Clone the repository:

```bash
git clone https://github.com/Saqibkhan-engineer/intern-performance.git
cd intern-performance
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/macOS
# .venv\Scripts\activate    # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the dashboard:

```bash
python app.py
```

Open the local Gradio URL shown in the terminal, typically:

```text
http://127.0.0.1:7860/
```

## Example Usage

A sample intern can be evaluated like this:

- Name: `Zoya Khan`
- Completion Time: `4 days`
- Feedback Rating: `4.2`
- Attendance: `88%`

After submitting, the app predicts a score and classifies the intern as either likely to excel or likely to struggle.

## Dependencies

The project relies on:

```txt
xgboost
pandas
```

The app also uses `gradio` and `spaces` for the user interface and deployment integration.

## Dataset

The repository includes a realistic intern dataset in CSV form for model development and evaluation. The dataset contains performance-related variables used to train the predictive model.

## Notes

- The model is designed for performance prediction using a small, structured feature set.
- The dashboard is built for demonstration and practical evaluation workflows.
- This project can be extended by adding more features like project quality, communication score, task completion rate, and mentor feedback.

## Potential Future Improvements

- Add more performance metrics
- Include model accuracy reporting
- Add charts and trends over time
- Deploy as a web application
- Save predictions to CSV or database
- Add login and admin dashboard features

## License

This project does not currently specify a license.

## Author

Saqibkhan-engineer

## Repository Summary

This project combines data science and application development to create a practical intern evaluation system. It predicts intern performance using measurable metrics and presents the results through an interactive dashboard that is simple to use and easy to extend.
