import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import time
import random
import subprocess
import zipfile
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ===== CONFIG =====
BASE_DIR = r"G:\My Drive\DMC_Competitions"
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions (File responses)")
TEST_DIR = os.path.join(BASE_DIR, "test_dataset")
GT_PATH = os.path.join(BASE_DIR, "ground_truth.csv")

LEADERBOARD_CSV = os.path.join(BASE_DIR, "leaderboard.csv")
LEADERBOARD_JSON = os.path.join(BASE_DIR, "leaderboard.json")

TIMEOUT_PER_IMAGE = 5

#some code
def fix_nested_folders(team_path):
    inner_items = os.listdir(team_path)

    if len(inner_items) == 1:
        inner_path = os.path.join(team_path, inner_items[0])

        if os.path.isdir(inner_path):
            for item in os.listdir(inner_path):
                shutil.move(
                    os.path.join(inner_path, item),
                    team_path
                )
            os.rmdir(inner_path)

# ===== LOAD GT =====
def load_ground_truth():
    gt_df = pd.read_csv(GT_PATH)
    return dict(zip(gt_df["image"], gt_df["label"]))

#unzipper
def extract_zip_submissions():
    for file in os.listdir(SUBMISSIONS_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(SUBMISSIONS_DIR, file)

            team_name = file.replace(".zip", "")
            extract_path = os.path.join(SUBMISSIONS_DIR, team_name)

            # Remove old extracted folder (if exists)
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)

            print(f"📦 Extracting {file}...")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                # ✅ Fix nested structure
                fix_nested_folders(extract_path)

                # ✅ DELETE ZIP HERE (important)
                os.remove(zip_path)
                print(f"🗑️ Deleted {file}")

            except Exception as e:
                print(f"❌ Failed to extract {file}:", e)

                
# ===== PREPROCESS =====
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
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
def evaluate_team(team, gt_dict, test_images):
    team_path = os.path.join(SUBMISSIONS_DIR, team)
    model_path = os.path.join(team_path, "model.pkl")

    print(f"\n🔍 Evaluating {team}")

    if not os.path.exists(model_path):
        print("❌ model.pkl missing")
        return float('inf')

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except:
        print("❌ model load failed")
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
    print(f"✅ {team} → {mse:.4f}")
    return mse

# ===== RUN FULL PIPELINE =====
def run_pipeline():
    print("\n⚙️ Running full evaluation pipeline...")
    extract_zip_submissions()
    try:
        gt_dict = load_ground_truth()
        test_images = list(gt_dict.keys())

        results = []
        print("📊 Results:", results)
        for team in os.listdir(SUBMISSIONS_DIR):
            if not os.path.isdir(os.path.join(SUBMISSIONS_DIR, team)):
                continue

            score = evaluate_team(team, gt_dict, test_images)
            results.append((team, score))

        # sort leaderboard
        leaderboard = sorted(results, key=lambda x: x[1])

        df = pd.DataFrame(leaderboard, columns=["Team", "MSE"])
        df.to_csv(LEADERBOARD_CSV, index=False)
        df.to_json(LEADERBOARD_JSON, orient="records")
        print("✅ JSON WRITTEN")
        print("📊 Leaderboard updated!")

        # ===== GIT PUSH =====
        print("📤 Pushing to GitHub...")
        subprocess.run(
            ["git", "pull", "origin", "main", "--no-rebase"],
            cwd=BASE_DIR
        )

        subprocess.run(
            ["git", "add", "leaderboard.json", "leaderboard.csv"],
            cwd=BASE_DIR
        )

        subprocess.run(
            ["git", "commit", "-m", "auto update leaderboard"],
            cwd=BASE_DIR,
            check=False
        )

        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=BASE_DIR
        )
        

        print("🚀 Live leaderboard updated!\n")

    except Exception as e:
        print("❌ Pipeline error:", e)


def check_and_run(team):
    team_path = os.path.join(SUBMISSIONS_DIR, team)
    model_path = os.path.join(team_path, "model.pkl")

    if os.path.exists(model_path):
        print(f"✅ Valid submission found: {team}")
        run_pipeline()
    else:
        print(f"⏳ Waiting for model.pkl in {team}")

# ===== AUTO LOOP (REPLACES WATCHER) =====
# ===== AUTO LOOP =====

CHECK_INTERVAL = 10  # seconds

def get_submission_state():
    state = []

    for root, dirs, files in os.walk(SUBMISSIONS_DIR):
        for name in files:
            state.append(os.path.join(root, name))

    return sorted(state)

last_state = None

if __name__ == "__main__":

    print("🔁 Starting auto polling system...")

    while True:
        try:
            extract_zip_submissions()   # 👈 ADD THIS LINE ALSO

            current_state = get_submission_state()

            print("📂 Current state:", current_state)

            if current_state != last_state:
                print("\n📦 Change detected!")
                run_pipeline()
                last_state = current_state.copy()
            else:
                print("⏳ No changes...")

        except Exception as e:
            print("❌ Error:", e)

        time.sleep(10)