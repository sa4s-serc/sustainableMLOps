import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
import numpy as np

# Paths to your main CSV files with prediction results
csv_files = [
    '/path/to/general_model_prediction_results_A1.csv',
    '/path/to/lstm_prediction_results_A2.csv',
    '/path/to/periodic_retraining_general_prediction_results_A3.csv',
    '/path/to/periodic_retraining_lstm_prediction_results_A4.csv',
    '/path/to/periodic_retraining_both_models_prediction_results_A5.csv',
    '/path/to/our_approach_prediction_results_A6.csv'
]

# Paths to additional CPU usage CSV files for the systems that were periodically retraining the models
additional_cpu_files = [
    '/path/to/periodic_retraining_general_retraining_results_A3.csv',
    '/path/to/periodic_retraining_lstm_retraining_results_A4.csv',
    '/path/to/periodic_retraining_both_models_retraining_results_A5.csv',
    '/path/to_our_approach_retraining_results_A6.csv'
]


# Labels for the x-axis
csv_labels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']

# Lists to hold R2 scores and CPU consumption
r2_scores = []
cpu_consumptions = []

# Processing CSV files
for idx, filepath in enumerate(csv_files):
    df = pd.read_csv(filepath)

    actual = df['Actual AQI']
    predicted = df['Predicted AQI']
    score = r2_score(actual, predicted)
    r2_scores.append(score)

    if idx < 2:
        avg_cpu = df['CPU Usage'].mean()
    else:
        additional_cpu_df = pd.read_csv(additional_cpu_files[idx - 2])
        avg_cpu = (additional_cpu_df[' CPU Consumption'].mean() + df['CPU Usage'].mean()) / 2
    cpu_consumptions.append(avg_cpu)
# Apply logarithmic scaling to CPU consumption
log_cpu_consumptions = np.log(cpu_consumptions)

# Create subplots
fig = make_subplots(specs=[[{'secondary_y': True}]])
positions = np.arange(len(csv_labels))  # Basic positions for each group
bar_width = 0.35  # Width of the bars

# Calculate the offset positions for R2 and CPU bars
r2_positions = positions - bar_width / 2
cpu_positions = positions + bar_width / 2

# Create the figure with secondary_y for CPU consumption
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Add R2 Score bars
fig.add_trace(go.Bar(
    x=r2_positions,
    y=r2_scores,
    name='R2 Score',
    marker=dict(color='#2449d1'),
    width=bar_width,
), secondary_y=False)

# Add CPU Consumption bars
fig.add_trace(go.Bar(
    x=cpu_positions,
    y=log_cpu_consumptions,
    name='Log of avg. CPU Consumption',
    marker=dict(color='#4ba658'),
    width=bar_width
), secondary_y=True)
# Set y-axes titles
fig.update_yaxes(title_text="<b>R2 Score </b>", secondary_y=False, color='#2449d1',title_font=dict(size=22),tickfont=dict(size=18))
fig.update_yaxes(title_text="<b>Log of avg. CPU Consumption</b>", secondary_y=True, color='#4ba658',title_font=dict(size=22),tickfont=dict(size=18))

# Customize x-axis ticks
fig.update_xaxes(
    tickvals=positions,  # Positioning ticks at the center of each group
    ticktext=csv_labels, # Custom label text
    tickfont=dict(size=18)  # Increase font size here

)

fig.update_layout(yaxis=dict(range=[0.8, 0.99]))
# Include custom legend entries without plotting them
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode='markers', name='A1: Linear, no RT', marker=dict(color='rgba(0,0,0,0)'))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode='markers', name='A2: LSTM, no RT', marker=dict(color='rgba(0,0,0,0)'))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode='markers', name='A3: Linear, periodic RT', marker=dict(color='rgba(0,0,0,0)'))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode='markers', name='A4: LSTM, periodic RT', marker=dict(color='rgba(0,0,0,0)'))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode='markers', name='A5: Both, periodic RT, switching',
               marker=dict(color='rgba(0,0,0,0)'))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode='markers', name='A6: Our Approach', marker=dict(color='rgba(0,0,0,0)'))
)

# Update layout for the legend and possibly other customizations
fig.update_layout(
    autosize=False,
    width=1300,
    height=800,
    legend=dict(
        x=0.01,
        y=0.99,
        traceorder='normal',
        font=dict(
            size=18.5,
        ),
    ),
    barmode='group',  # This ensures bars are grouped
)

fig.write_image("images/fig1.pdf", scale=2)
fig.show()
