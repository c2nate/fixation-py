import pandas as pd
import numpy as np

# Load gaze data
gaze_data = pd.read_csv("gaze_product_data.csv")

# Convert timeElapsed to numeric (handles potential formatting issues)
gaze_data['timeElapsed'] = pd.to_numeric(gaze_data['timeElapsed'], errors='coerce')

# Drop rows with missing coordinates or time
gaze_data = gaze_data.dropna(subset=['x', 'y', 'timeElapsed'])

# Fixation Detection using I-DT Algorithm
def detect_fixations(gaze_data, threshold=30, min_duration=100):
    fixations = []
    start_idx = 0
    gaze_np = gaze_data[['x', 'y', 'timeElapsed']].to_numpy()
    titles = gaze_data['Title'].tolist()  # Get corresponding product titles

    while start_idx < len(gaze_np):
        end_idx = start_idx
        while (end_idx < len(gaze_np) and 
               (np.max(gaze_np[start_idx:end_idx+1, 0]) - np.min(gaze_np[start_idx:end_idx+1, 0]) < threshold) and
               (np.max(gaze_np[start_idx:end_idx+1, 1]) - np.min(gaze_np[start_idx:end_idx+1, 1]) < threshold)):
            end_idx += 1
        
        duration = gaze_np[end_idx-1, 2] - gaze_np[start_idx, 2]
        if duration >= min_duration:
            fixations.append({
                "x": np.mean(gaze_np[start_idx:end_idx, 0]),
                "y": np.mean(gaze_np[start_idx:end_idx, 1]),
                "duration": duration,
                "Title": titles[start_idx]  # Assign fixation to the corresponding product title
            })
        
        start_idx = end_idx  # Move to next fixation window

    return pd.DataFrame(fixations)

# Compute Fixations
fixation_df = detect_fixations(gaze_data)

# Save Fixations to CSV
fixation_df.to_csv("fixation_features.csv", index=False)

print("Fixation Features Extracted:")
print(fixation_df.head())
