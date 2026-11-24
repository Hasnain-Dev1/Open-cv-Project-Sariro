from pathlib import Path
import cv2
import numpy as np

KNOWN_DIR = Path("known_faces")


def _load_face_cascade() -> cv2.CascadeClassifier:
    """Load the frontal face Haar cascade safely across OpenCV builds."""
    filename = "haarcascade_frontalface_default.xml"
    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)

    candidates = []
    if cascade_dir:
        candidates.append(Path(cascade_dir) / filename)
    candidates.append(Path(cv2.__file__).resolve().parent / "data" / filename)

    for path in candidates:
        if path.exists():
            clf = cv2.CascadeClassifier(str(path))
            if not clf.empty():
                return clf

    raise FileNotFoundError(f"Could not load {filename} from OpenCV data paths.")


FACE_CASCADE = _load_face_cascade()


def _match_histogram(
    face_gray: np.ndarray, known_hists: list[tuple[str, np.ndarray]]
) -> tuple[str | None, float]:
    """Compare a grayscale face crop against saved reference histograms."""
    resized = cv2.resize(face_gray, (100, 100))
    hist = cv2.calcHist([resized], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)

    best_name = None
    best_score = -1.0
    for name, ref_hist in known_hists:
        score = float(cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CORREL))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def load_known_profiles(known_dir: Path = KNOWN_DIR) -> list[tuple[str, np.ndarray]]:
    """Load reference images from known_faces/ and build normalized histograms."""
    known_dir.mkdir(exist_ok=True)
    profiles: list[tuple[str, np.ndarray]] = []

    for img_path in sorted(known_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) > 0:
            x, y, w, h = faces[0]
            crop = cv2.resize(gray[y : y + h, x : x + w], (100, 100))
        else:
            crop = cv2.resize(gray, (100, 100))

        hist = cv2.calcHist([crop], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        profiles.append((img_path.stem, hist))

    return profiles


def analyze_hosted_photo(
    bgr_frame,
    known_profiles: list[tuple[str, np.ndarray]],
    match_threshold: float = 0.75,
) -> tuple[str, object]:
    """Draw boxes around faces with labels underneath; return (status, annotated_bgr)."""
    annotated = bgr_frame.copy()
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    h_img, w_img = bgr_frame.shape[:2]

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(50, 50),
    )

    if len(faces) == 0:
        cv2.rectangle(annotated, (20, 20), (w_img - 20, h_img - 20), (0, 0, 255), 3)
        cv2.putText(
            annotated,
            "Bad Guy (Covered Face)",
            (30, min(h_img - 30, 60)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
        )
        return "COVERED", annotated

    overall_status = "UNKNOWN"

    for (x, y, w, h) in faces:
        x2 = min(w_img - 1, x + w)
        y2 = min(h_img - 1, y + h)
        face_gray = gray[y:y2, x:x2]

        label = "Unknown Face"
        color = (0, 165, 255)  # Orange in BGR

        if face_gray.size > 0 and known_profiles:
            name, score = _match_histogram(face_gray, known_profiles)
            if name and score >= match_threshold:
                label = f"Good Guy ({name})"
                color = (0, 180, 0)  # Green in BGR
                if overall_status != "COVERED":
                    overall_status = "GOOD_GUY"

        cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
        text_x = max(5, x)
        text_y = min(h_img - 10, y2 + 24)
        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    return overall_status, annotated