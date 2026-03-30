import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import time
import random

# ===== CONFIG =====
BASE_DIR = r"G:\My Drive\DMC_Competitions"
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
TEST_DIR = os.path.join(BASE_DIR, "test_dataset")
GT_PATH = os.path.join(BASE_DIR, "ground_truth.csv")

LEADERBOARD_CSV = os.path.join(BASE_DIR, "leaderboard.csv")
LEADERBOARD_JSON = os.path.join(BASE_DIR, "leaderboard.json")

TIMEOUT_PER_IMAGE = 5

# ===== LOAD GT =====
gt_df = pd.read_csv(GT_PATH)
gt_dict = dict(zip(gt_df["image"], gt_df["label"]))
test_images = list(gt_dict.keys())

# ===== PREPROCESS =====
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    return img.flatten().reshape(1, -1)

# ===== SAFE PREDICT =====
def safe_predict(model, X):
    start = time.time()
    prob = model.predict_proba(X)[0][1]
    if time.time() - start > TIMEOUT_PER_IMAGE:
        raise TimeoutError
    return prob

# ===== EVALUATE TEAM =====
def evaluate_team(team):
    team_path = os.path.join(SUBMISSIONS_DIR, team)
    model_path = os.path.join(team_path, "model.pkl")

    print(f"\nEvaluating {team}")

    if not os.path.exists(model_path):
        print("model.pkl missing")
        return float('inf')

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except:
        print("model load failed")
        return float('inf')

    preds, gts = [], []
    random.shuffle(test_images)

    for img_name in test_images:
        img_path = os.path.join(TEST_DIR, img_name)

        try:
            X = preprocess(img_path)
            prob = safe_predict(model, X)

            if not (0 <= prob <= 1):
                raise ValueError

        except:
            prob = 0.0

        preds.append(prob)
        gts.append(gt_dict[img_name])

    mse = float(np.mean((np.array(preds) - np.array(gts)) ** 2))
    print(f"{team} → {mse:.4f}")
    return mse

# ===== MAIN EVALUATION =====
results = []

for team in os.listdir(SUBMISSIONS_DIR):
    if not os.path.isdir(os.path.join(SUBMISSIONS_DIR, team)):
        continue

    score = evaluate_team(team)
    results.append((team, score))

# ===== SORT =====
leaderboard = sorted(results, key=lambda x: x[1])

# ===== SAVE CSV =====
df = pd.DataFrame(leaderboard, columns=["Team", "MSE"])
df.to_csv(LEADERBOARD_CSV, index=False)

# ===== SAVE JSON =====
df.to_json(LEADERBOARD_JSON, orient="records")

print("\nLeaderboard generated!")

# ===== AUTO GIT PUSH =====
print("Pushing to GitHub...")

os.chdir(BASE_DIR)

os.system("git add .")
os.system('git commit -m "auto update leaderboard"')
os.system("git push")

print("Done 🚀")