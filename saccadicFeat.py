import pandas as pd
import numpy as np

# Load fixation data
fixation_df = pd.read_csv("fixation_features.csv")

def compute_saccades(fixation_df, saccade_threshold=40):
    saccades = []
    
    for i in range(len(fixation_df) - 1):
        x1, y1, t1, title1 = fixation_df.iloc[i]['x'], fixation_df.iloc[i]['y'], fixation_df.iloc[i]['duration'], fixation_df.iloc[i]['Title']
        x2, y2, t2, title2 = fixation_df.iloc[i+1]['x'], fixation_df.iloc[i+1]['y'], fixation_df.iloc[i+1]['duration'], fixation_df.iloc[i+1]['Title']

        # Compute Euclidean distance
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Compute velocity (distance / time difference)
        time_diff = t2 - t1
        velocity = distance / time_diff if time_diff > 0 else 0
        
        # Compute saccadic direction (angle in radians)
        direction = np.arctan2(y2 - y1, x2 - x1)

        # Classify as a saccade if distance exceeds the threshold
        if distance > saccade_threshold:
            saccades.append({
                "start_x": x1, "start_y": y1,
                "end_x": x2, "end_y": y2,
                "distance": distance,
                "velocity": velocity,
                "direction": direction,  # In radians
                "from_title": title1,  # Product user moved from
                "to_title": title2  # Product user moved to
            })

    return pd.DataFrame(saccades)

# Compute Saccades
saccade_df = compute_saccades(fixation_df)

# Save to CSV
saccade_df.to_csv("saccadic_features.csv", index=False)

print("Saccadic Features Extracted:")
print(saccade_df.head())
