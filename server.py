import base64
import gc
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template_string

# Prevent OpenCV from consuming excessive CPU threads
cv2.setNumThreads(2)

# Global Configuration
TARGET_W, TARGET_H = 600, 825
CACHE_FILE = "pokemon_index_cache.pkl"

app = Flask(__name__)


def process_single_card(path_str):
    """Worker function executed in parallel across CPU cores for indexing."""
    p = Path(path_str)
    parts = p.parts
    if len(parts) < 4:
        return None

    filename = p.name
    if "_low" in filename:
        return None

    lang = parts[-3]
    extension_code = parts[-2]
    card_num = filename.split(".")[0].rsplit("_", 1)[0]

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img_resized = cv2.resize(img, (TARGET_W, TARGET_H))

    orb = cv2.ORB_create(nfeatures=500)
    _, des = orb.detectAndCompute(img_resized, None)

    if des is None:
        return None

    metadata = {
        "language": lang,
        "set": extension_code,
        "number": card_num,
        "path": str(p),
    }

    return (metadata, des)


class PokemonCardScannerApp:
    def __init__(self):
        self.target_w = TARGET_W
        self.target_h = TARGET_H
        self.orb = cv2.ORB_create(nfeatures=500)
        self.descriptor_map = []
        self.metadata_db = []
        self.flann = None

    def build_or_load_index(self, root_path="pokemon_cards"):
        """Loads cached index from disk or generates FLANN tree."""
        if os.path.exists(CACHE_FILE):
            print(f"⚡ Loading cached index from '{CACHE_FILE}'...")
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)

            self.metadata_db = data["metadata_db"]
            self.descriptor_map = data["descriptor_map"]
            stacked_descriptors = data["stacked_descriptors"]

            del data
            gc.collect()

            self._build_flann_tree(stacked_descriptors)
            print(f"✅ Ready! Loaded {len(self.metadata_db)} cards.")
            return

        print("🔍 First run detected. Scanning dataset directory...")
        root = Path(root_path)
        image_paths = [str(p) for p in root.glob("*/*/*_high.png")]

        max_workers = min(4, os.cpu_count() or 1)
        all_descriptors = []
        card_counter = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_card, p) for p in image_paths]

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                meta, des = result
                self.metadata_db.append(meta)
                all_descriptors.append(des)

                for _ in range(len(des)):
                    self.descriptor_map.append(card_counter)

                card_counter += 1

        stacked_descriptors = np.vstack(all_descriptors)
        del all_descriptors
        gc.collect()

        cache_data = {
            "metadata_db": self.metadata_db,
            "descriptor_map": self.descriptor_map,
            "stacked_descriptors": stacked_descriptors,
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        del cache_data
        gc.collect()

        self._build_flann_tree(stacked_descriptors)

    def _build_flann_tree(self, stacked_descriptors):
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.flann.add([stacked_descriptors])
        self.flann.train()

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_and_warp(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and cv2.contourArea(c) > 10000:
                pts = approx.reshape(4, 2)
                rect = self.order_points(pts)

                dst = np.array(
                    [
                        [0, 0],
                        [self.target_w - 1, 0],
                        [self.target_w - 1, self.target_h - 1],
                        [0, self.target_h - 1],
                    ],
                    dtype="float32",
                )

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame, M, (self.target_w, self.target_h))
                return warped

        # If strict contour detection fails, fallback to resizing the entire frame
        return cv2.resize(frame, (self.target_w, self.target_h))

    def identify_card(self, cropped_card, min_matches=12):
        gray = cv2.cvtColor(cropped_card, cv2.COLOR_BGR2GRAY)
        _, des_query = self.orb.detectAndCompute(gray, None)

        if des_query is None or len(self.metadata_db) == 0:
            return None, None, 0.0

        matches = self.flann.knnMatch(des_query, k=2)
        card_match_counts = {}

        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                card_idx = self.descriptor_map[m.trainIdx]
                card_match_counts[card_idx] = card_match_counts.get(card_idx, 0) + 1

        if not card_match_counts:
            return None, None, 0.0

        best_card_idx = max(card_match_counts, key=card_match_counts.get)
        max_matches = card_match_counts[best_card_idx]

        meta = self.metadata_db[best_card_idx]
        ref_image = cv2.imread(meta["path"])
        if ref_image is not None:
            ref_image = cv2.resize(ref_image, (self.target_w, self.target_h))

        confidence = min(100.0, round((max_matches / 28.0) * 100, 1))

        if max_matches >= min_matches:
            return meta, ref_image, confidence

        return meta, ref_image, confidence


# Initialize backend scanner engine
scanner = PokemonCardScannerApp()
scanner.build_or_load_index("pokemon_cards")

# Embedded Mobile-Optimized HTML Frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pokémon Card Scanner</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
        h1 { font-size: 1.5rem; margin-bottom: 20px; }
        .upload-btn { background: #007bff; color: #fff; padding: 15px 25px; border-radius: 8px; font-size: 1.1rem; cursor: pointer; display: inline-block; margin-bottom: 20px; font-weight: bold; }
        input[type="file"] { display: none; }
        #status { font-size: 1rem; color: #ffca28; margin-bottom: 15px; }
        #result-container { display: none; margin-top: 20px; background: #1e1e1e; padding: 15px; border-radius: 12px; }
        .meta-card { background: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 15px; }
        .img-grid { display: flex; justify-content: space-around; gap: 10px; margin-top: 10px; }
        .img-box { flex: 1; }
        .img-box img { width: 100%; border-radius: 6px; }
        .img-label { font-size: 0.8rem; color: #aaa; margin-bottom: 5px; }
    </style>
</head>
<body>
    <h1>🎴 Mobile Card Scanner</h1>
    
    <label class="upload-btn">
        📷 Take Picture / Upload
        <input type="file" accept="image/*" capture="environment" id="file-input" onchange="processImage(this)">
    </label>

    <div id="status"></div>

    <div id="result-container">
        <div class="meta-card">
            <h2 id="card-title" style="margin:0 0 5px 0; font-size: 1.2rem; color: #4caf50;"></h2>
            <div id="card-conf" style="font-size: 0.9rem; color: #ccc;"></div>
        </div>
        <div class="img-grid">
            <div class="img-box">
                <div class="img-label">Captured</div>
                <img id="scanned-img" src="" alt="Scanned">
            </div>
            <div class="img-box">
                <div class="img-label">Matched Database</div>
                <img id="ref-img" src="" alt="Matched">
            </div>
        </div>
    </body>
    <script>
        async function processImage(input) {
            if (!input.files || !input.files[0]) return;

            const status = document.getElementById("status");
            const resultBox = document.getElementById("result-container");
            status.innerText = "⏳ Processing image on PC...";
            resultBox.style.display = "none";

            const formData = new FormData();
            formData.append("image", input.files[0]);

            try {
                const response = await fetch("/scan", { method: "POST", body: formData });
                const data = await response.json();

                if (!response.ok) {
                    status.innerText = "❌ " + (data.error || "Processing failed.");
                    return;
                }

                status.innerText = "";
                document.getElementById("card-title").innerText = data.card_title;
                document.getElementById("card-conf").innerText = `Confidence: ${data.confidence}%`;
                document.getElementById("scanned-img").src = data.scanned_img;
                document.getElementById("ref-img").src = data.ref_img;
                resultBox.style.display = "block";
            } catch (err) {
                status.innerText = "❌ Network error or request timed out.";
            }
        }
    </script>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Perspective warp / extract card portion
    warped_card = scanner.detect_and_warp(frame)

    # Perform FLANN feature matching
    meta, ref_img, confidence = scanner.identify_card(warped_card)

    # Encode images to Base64 data URLs for JSON transport
    _, scanned_buf = cv2.imencode(".jpg", warped_card)
    scanned_b64 = "data:image/jpeg;base64," + base64.b64encode(scanned_buf).decode("utf-8")

    if ref_img is None:
        ref_img = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        cv2.putText(
            ref_img,
            "No Match",
            (50, TARGET_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    _, ref_buf = cv2.imencode(".jpg", ref_img)
    ref_b64 = "data:image/jpeg;base64," + base64.b64encode(ref_buf).decode("utf-8")

    card_title = (
        f"{meta['set'].upper()} #{meta['number']} ({meta['language'].upper()})"
        if meta
        else "Unknown Card"
    )

    return jsonify(
        {
            "card_title": card_title,
            "confidence": confidence,
            "scanned_img": scanned_b64,
            "ref_img": ref_b64,
        }
    )


if __name__ == "__main__":
    # Required dependencies: install flask via `pip install flask`
    # Bound to 0.0.0.0 to expose the server to your local Wi-Fi network
    app.run(host="0.0.0.0", port=5000, debug=False)