import gc
import os
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np

# Prevent OpenCV from consuming excessive CPU threads
cv2.setNumThreads(2)

# Global Configuration
TARGET_W, TARGET_H = 350, 490
CACHE_FILE = "pokemon_index_cache.pkl"


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

        # Query ORB detector for live frames
        self.orb = cv2.ORB_create(nfeatures=500)

        # Dataset storage
        self.descriptor_map = []
        self.metadata_db = []
        self.flann = None

        # Threading & Loading state
        self.is_processing = False
        self.match_result = None
        self.avg_match_time = 1.2  # Initial estimated duration in seconds

    # -------------------------------------------------------------------------
    # INDEXING & DISK CACHING
    # -------------------------------------------------------------------------
    def build_or_load_index(self, root_path="pokemon_cards"):
        """Loads cached index from disk if present, else runs multi-core indexing."""
        if os.path.exists(CACHE_FILE):
            print(f"⚡ Loading cached index from '{CACHE_FILE}'...")
            start_t = time.time()
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)

            self.metadata_db = data["metadata_db"]
            self.descriptor_map = data["descriptor_map"]
            stacked_descriptors = data["stacked_descriptors"]

            del data
            gc.collect()

            self._build_flann_tree(stacked_descriptors)
            print(
                f"✅ Ready! Loaded {len(self.metadata_db)} cards in {time.time() - start_t:.2f}s."
            )
            return

        print("🔍 First run detected. Scanning dataset directory...")
        root = Path(root_path)
        image_paths = [str(p) for p in root.glob("*/*/*.png")]

        # Throttled worker count to prevent Linux OOM system crashes
        max_workers = min(4, os.cpu_count() or 1)
        print(
            f"🚀 Parallel indexing {len(image_paths)} candidate files across {max_workers} CPU workers..."
        )

        start_t = time.time()
        all_descriptors = []
        card_counter = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_card, p) for p in image_paths
            ]

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

        print(
            f"⚡ Processing complete in {time.time() - start_t:.2f}s. Stacking descriptors..."
        )
        stacked_descriptors = np.vstack(all_descriptors)

        del all_descriptors
        gc.collect()

        print(f"💾 Saving index to '{CACHE_FILE}'...")
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
        """Constructs FLANN LSH tree for fast approximate lookup."""
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

    # -------------------------------------------------------------------------
    # STAGE 1: CARD DETECTION & PERSPECTIVE WARP
    # -------------------------------------------------------------------------
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

        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
                warped = cv2.warpPerspective(
                    frame, M, (self.target_w, self.target_h)
                )
                return warped, rect

        return None, None

    # -------------------------------------------------------------------------
    # STAGE 2: IDENTIFY CARD & ASYNC WORKER
    # -------------------------------------------------------------------------
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
                card_match_counts[card_idx] = (
                    card_match_counts.get(card_idx, 0) + 1
                )

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

    def process_card_async(self, cropped_card):
        """Worker target for background thread during picture processing."""
        self.is_processing = True
        t0 = time.time()

        meta, ref_img, conf = self.identify_card(cropped_card)

        elapsed = time.time() - t0
        self.avg_match_time = max(0.4, elapsed)  # Update rolling time estimate
        self.match_result = (meta, ref_img, conf, elapsed)
        self.is_processing = False

    # -------------------------------------------------------------------------
    # UI RENDERERS: LOADING SCREEN & COMPARISON VIEW
    # -------------------------------------------------------------------------
    def render_loading_screen(self, base_card_img, elapsed_time):
        """Creates an animated loading window with progress bar and countdown."""
        overlay = base_card_img.copy()
        h, w, _ = overlay.shape

        # Dim captured image background
        dark_layer = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = cv2.addWeighted(overlay, 0.3, dark_layer, 0.7, 0)

        # Estimate remaining time
        est_total = max(0.5, self.avg_match_time)
        progress = min(1.0, elapsed_time / est_total)
        rem_time = max(0.0, est_total - elapsed_time)

        # Draw animated spinner
        center = (w // 2, h // 2 - 40)
        angle = int((time.time() * 360) % 360)
        cv2.ellipse(
            overlay,
            center,
            (35, 35),
            0,
            angle,
            angle + 270,
            (0, 255, 255),
            4,
        )

        # Draw progress bar
        bar_w = int(w * 0.75)
        bar_h = 16
        x1 = (w - bar_w) // 2
        y1 = h // 2 + 30

        cv2.rectangle(
            overlay, (x1, y1), (x1 + bar_w, y1 + bar_h), (50, 50, 50), -1
        )
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x1 + int(bar_w * progress), y1 + bar_h),
            (0, 255, 0),
            -1,
        )
        cv2.rectangle(
            overlay, (x1, y1), (x1 + bar_w, y1 + bar_h), (200, 200, 200), 1
        )

        # Status text
        cv2.putText(
            overlay,
            "MATCHING CARD...",
            (w // 2 - 80, h // 2 - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Elapsed: {elapsed_time:.1f}s",
            (x1, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            overlay,
            f"Est. left: ~{rem_time:.1f}s",
            (x1 + bar_w - 110, y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        return overlay

    def create_comparison_display(
        self, scanned_crop, ref_image, metadata, confidence, match_time
    ):
        if ref_image is None:
            ref_image = np.zeros(
                (self.target_h, self.target_w, 3), dtype=np.uint8
            )
            cv2.putText(
                ref_image,
                "No Match",
                (50, self.target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        divider = np.zeros((self.target_h, 10, 3), dtype=np.uint8)
        side_by_side = np.hstack((scanned_crop, divider, ref_image))

        header = np.zeros((80, side_by_side.shape[1], 3), dtype=np.uint8)

        cv2.putText(
            header,
            "SCANNED IMAGE",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            header,
            "TOP MATCH IN DATABASE",
            (self.target_w + 30, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if metadata:
            color = (0, 255, 0) if confidence >= 40.0 else (0, 165, 255)
            card_title = f"{metadata['set'].upper()} #{metadata['number']} ({metadata['language'].upper()})"
            conf_title = f"Conf: {confidence}% | Time: {match_time:.2f}s"

            cv2.putText(
                header,
                card_title,
                (self.target_w + 30, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            cv2.putText(
                header,
                conf_title,
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return np.vstack((header, side_by_side))


# -----------------------------------------------------------------------------
# APPLICATION ENTRY POINT & WEBCAM LOOP
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = PokemonCardScannerApp()

    # 1. Load or build index
    app.build_or_load_index("pokemon_cards")

    # 2. Open live camera feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        exit()

    print("\n🎥 Webcam feed active!")
    print("👉 Align card and press [SPACE] to take a picture and scan.")
    print("👉 Press [Q] to quit.")

    app_state = "LIVE"  # States: "LIVE", "PROCESSING", "RESULT"
    captured_warped = None
    start_proc_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break

        warped_card, quad_corners = app.detect_and_warp(frame)

        # ---------------------------------------------------------------------
        # STATE 1: LIVE WEBCAM FEED
        # ---------------------------------------------------------------------
        if app_state == "LIVE":
            if warped_card is not None:
                cv2.polylines(
                    frame, [quad_corners.astype(int)], True, (0, 255, 0), 3
                )
                cv2.putText(
                    frame,
                    "CARD READY! Press [SPACE] to capture",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if key == ord(" "):
                    captured_warped = warped_card.copy()
                    start_proc_time = time.time()
                    app_state = "PROCESSING"

                    # Trigger matching in background thread
                    t = threading.Thread(
                        target=app.process_card_async, args=(captured_warped,)
                    )
                    t.daemon = True
                    t.start()
            else:
                cv2.putText(
                    frame,
                    "Position card in camera view...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Main Webcam Feed", frame)

        # ---------------------------------------------------------------------
        # STATE 2: PROCESSING / LOADING OVERLAY
        # ---------------------------------------------------------------------
        elif app_state == "PROCESSING":
            cv2.putText(
                frame,
                "Processing image...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Main Webcam Feed", frame)

            elapsed = time.time() - start_proc_time
            loading_view = app.render_loading_screen(captured_warped, elapsed)
            cv2.imshow("Card Scanner (Live vs Database)", loading_view)

            if not app.is_processing:
                app_state = "RESULT"

        # ---------------------------------------------------------------------
        # STATE 3: DISPLAY RESULTS
        # ---------------------------------------------------------------------
        elif app_state == "RESULT":
            cv2.putText(
                frame,
                "Press [SPACE] to capture new card",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Main Webcam Feed", frame)

            meta, ref_img, confidence, match_time = app.match_result
            comparison_view = app.create_comparison_display(
                captured_warped, ref_img, meta, confidence, match_time
            )
            cv2.imshow("Card Scanner (Live vs Database)", comparison_view)

            if key == ord(" "):
                app_state = "LIVE"

    cap.release()
    cv2.destroyAllWindows()