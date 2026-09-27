import gradio as gr
import pandas as pd
from xgboost import XGBRegressor
import spaces


@spaces.GPU(duration=1)
def _dummy_gpu():
    return None


model = XGBRegressor()
model.load_model('intern_performance_model.json')

initial_data = pd.DataFrame({
    "Intern_Name": ["Ali Khan", "Sara Ahmed", "Usman Tariq"],
    "Completion_Time": [3, 2, 6],
    "Feedback_Rating": [4.5, 4.8, 2.5],
    "Attendance": [92.0, 98.0, 60.0]
})


def get_prediction(time, feedback, attendance):
    time = float(time)
    feedback = float(feedback)
    attendance = float(attendance)

    df = pd.DataFrame({
        'Completion_Time': [time],
        'Feedback_Rating': [feedback],
        'Attendance': [attendance]
    })

    score = float(model.predict(df)[0])

    if score >= 69.0:
        status = "🌟 Likely to Excel"
    else:
        status = "⚠️ Likely to Struggle"

    return round(score, 2), status


def load_initial_dashboard():
    dash_data = initial_data.copy()
    scores = []
    statuses = []
    for _, row in dash_data.iterrows():
        s, st = get_prediction(row['Completion_Time'], row['Feedback_Rating'], row['Attendance'])
        scores.append(s)
        statuses.append(st)

    dash_data['Predicted_Score'] = scores
    dash_data['Performance_Status'] = statuses
    return dash_data


def add_new_intern(name, time, feedback, attendance, current_dataframe):
    score, status = get_prediction(time, feedback, attendance)

    new_row = pd.DataFrame({
        "Intern_Name": [name],
        "Completion_Time": [float(time)],
        "Feedback_Rating": [float(feedback)],
        "Attendance": [float(attendance)],
        "Predicted_Score": [score],
        "Performance_Status": [status]
    })

    updated_df = pd.concat([current_dataframe, new_row], ignore_index=True)

    return updated_df, updated_df


with gr.Blocks() as dashboard:
    gr.Markdown("# Intern Performance Analytics Dashboard")
    gr.Markdown("Track intern performance and predict future outcomes using an XGBoost machine learning model")

    # Data store karne ke liye hidden state
    current_data_state = gr.State(load_initial_dashboard())

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ➕ Add New Intern")

            name_input = gr.Textbox(label="Intern Name", placeholder="e.g. Zoya...")

            time_input = gr.Dropdown(choices=[1, 2, 3, 4, 5, 6, 7], label="Completion Time (Days)", value=4)
            feedback_input = gr.Slider(minimum=1.0, maximum=5.0, step=0.1, label="Feedback Rating", value=3.5)
            attendance_input = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, label="Attendance (%)", value=80.0)

            add_btn = gr.Button("Predict & Add to Dashboard", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📈 All Interns Record")

            data_table = gr.Dataframe(
                value=load_initial_dashboard(),
                headers=["Intern_Name", "Completion_Time", "Feedback_Rating", "Attendance", "Predicted_Score",
                         "Performance_Status"],
                interactive=False
            )

    add_btn.click(
        fn=add_new_intern,
        inputs=[name_input, time_input, feedback_input, attendance_input, current_data_state],
        outputs=[current_data_state, data_table]
    )

if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Soft(), ssr_mode=False)