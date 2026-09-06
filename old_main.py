import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np

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
    # Skip low-res versions if high quality exists
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

            self._build_flann_tree(stacked_descriptors)
            print(
                f"✅ Ready! Loaded {len(self.metadata_db)} cards in {time.time() - start_t:.2f}s."
            )
            return

        print("🔍 First run detected. Scanning dataset directory...")
        root = Path(root_path)
        image_paths = [str(p) for p in root.glob("*/*/*.png")]
        print(
            f"🚀 Parallel indexing {len(image_paths)} candidate files across CPU cores..."
        )

        start_t = time.time()
        all_descriptors = []
        card_counter = 0

        with ProcessPoolExecutor() as executor:
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

        print(f"💾 Saving index to '{CACHE_FILE}'...")
        cache_data = {
            "metadata_db": self.metadata_db,
            "descriptor_map": self.descriptor_map,
            "stacked_descriptors": stacked_descriptors,
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    # STAGE 2: IDENTIFY TOP MATCH (LAZY LOAD IMAGE FROM DISK)
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
        # Lazy Load image off disk only when identified
        ref_image = cv2.imread(meta["path"])
        if ref_image is not None:
            ref_image = cv2.resize(ref_image, (self.target_w, self.target_h))

        confidence = min(100.0, round((max_matches / 28.0) * 100, 1))

        if max_matches >= min_matches:
            return meta, ref_image, confidence

        return meta, ref_image, confidence

    # -------------------------------------------------------------------------
    # STAGE 3: SIDE-BY-SIDE DISPLAY
    # -------------------------------------------------------------------------
    def create_comparison_display(
        self, scanned_crop, ref_image, metadata, confidence
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

        header = np.zeros((70, side_by_side.shape[1], 3), dtype=np.uint8)

        cv2.putText(
            header,
            "SCANNED",
            (20, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            header,
            "TOP MATCH IN DATABASE",
            (self.target_w + 30, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        if metadata:
            color = (0, 255, 0) if confidence >= 40.0 else (0, 165, 255)
            card_title = f"{metadata['set'].upper()} #{metadata['number']} ({metadata['language'].upper()})"
            conf_title = f"Conf: {confidence}%"

            cv2.putText(
                header,
                card_title,
                (self.target_w + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                header,
                conf_title,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return np.vstack((header, side_by_side))


# -----------------------------------------------------------------------------
# APPLICATION ENTRY POINT & WEBCAM LOOP
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = PokemonCardScannerApp()

    # 1. Build or load pre-compiled index cache
    app.build_or_load_index("pokemon_cards")

    # 2. Open live camera feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        exit()

    print("\n🎥 Webcam feed active! Place a Pokemon card in view.")
    print("Press 'q' in any video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect card in live camera frame
        warped_card, quad_corners = app.detect_and_warp(frame)

        if warped_card is not None:
            # Draw tracking boundary on webcam video feed
            cv2.polylines(
                frame, [quad_corners.astype(int)], True, (0, 255, 0), 3
            )

            # Match against database
            meta, ref_img, confidence = app.identify_card(warped_card)

            # Draw side-by-side comparison window
            comparison_view = app.create_comparison_display(
                warped_card, ref_img, meta, confidence
            )
            cv2.imshow("Card Scanner (Live vs Database)", comparison_view)
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()