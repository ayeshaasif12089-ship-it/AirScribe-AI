"""
Project: Draw Without Touching
--------------------------------
Draw in the air using just your index finger, tracked live through
your webcam with MediaPipe hand landmarks + OpenCV.

CONTROLS (all gesture based, no keyboard needed while drawing):
  - Index finger only up      -> Draw mode (draws where your fingertip moves)
  - Index + middle finger up  -> Selection mode (move freely, pick colors from
                                  the top bar without drawing)
  - All 5 fingers up          -> Pause (hover, nothing happens)
  - Fist (no fingers up)      -> Pause (hover, nothing happens)

KEYBOARD SHORTCUTS:
  - 'c'  -> clear the canvas
  - 's'  -> save your drawing as a PNG
  - 'q'  -> quit

Requirements:
  pip install opencv-python mediapipe numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 60
SMOOTHING = 5  # higher = smoother but more "lag" on the line

# Color palette (name, BGR color). "Eraser" is handled specially.
PALETTE = [
    ("Red",    (0, 0, 255)),
    ("Green",  (0, 220, 0)),
    ("Blue",   (255, 0, 0)),
    ("Yellow", (0, 220, 220)),
    ("Purple", (200, 0, 200)),
    ("Eraser", (0, 0, 0)),      # special-cased below
]
SWATCH_W = CAM_WIDTH // len(PALETTE)
SWATCH_H = 90


# ---------------------------------------------------------------------------
# HAND TRACKING HELPERS
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky landmark IDs


def fingers_up(hand_landmarks, handedness_label):
    """
    Returns a list of 5 booleans (thumb..pinky) telling which fingers are
    extended, based on landmark y (and x for thumb) positions.
    """
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: compare x-coordinates (mirrored image, so logic depends on hand)
    if handedness_label == "Right":
        fingers.append(lm[TIP_IDS[0]].x < lm[TIP_IDS[0] - 1].x)
    else:
        fingers.append(lm[TIP_IDS[0]].x > lm[TIP_IDS[0] - 1].x)

    # Other 4 fingers: tip above the pip joint (y is smaller near top of frame)
    for tip_id in TIP_IDS[1:]:
        fingers.append(lm[tip_id].y < lm[tip_id - 2].y)

    return fingers  # [thumb, index, middle, ring, pinky] as booleans


def draw_palette(img, selected_color_name):
    """Draws the color selection bar at the top of the frame."""
    for i, (name, color) in enumerate(PALETTE):
        x1, x2 = i * SWATCH_W, (i + 1) * SWATCH_W
        swatch_color = (40, 40, 40) if name == "Eraser" else color
        cv2.rectangle(img, (x1, 0), (x2, SWATCH_H), swatch_color, -1)

        # highlight the currently selected swatch
        if name == selected_color_name:
            cv2.rectangle(img, (x1, 0), (x2, SWATCH_H), (255, 255, 255), 4)

        text_color = (255, 255, 255) if name == "Eraser" else (0, 0, 0)
        cv2.putText(img, name, (x1 + 10, SWATCH_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check your camera connection/permissions.")
        return

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera negotiated resolution: {int(actual_w)}x{int(actual_h)}")

    # --- Warm-up: DSHOW sometimes needs a moment before frames are ready ---
    warm_up_ok = False
    for attempt in range(30):  # try for up to ~3 seconds
        success, _ = cap.read()
        if success:
            warm_up_ok = True
            break
        time.sleep(0.1)

    if not warm_up_ok:
        print("ERROR: Camera opened but never produced a frame.")
        print("Try: closing other apps using the camera, unplugging/replugging an external webcam,")
        print("or check Windows camera privacy settings (Settings > Privacy > Camera).")
        cap.release()
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # Canvas that persists across frames (this is where the drawing lives)
    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)

    selected_name, selected_color = PALETTE[0]  # default: Red
    prev_x, prev_y = 0, 0
    smooth_x, smooth_y = 0, 0

    prev_time = 0

    os.makedirs("drawings", exist_ok=True)

    print("Starting... press 'q' in the video window to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: dropped a frame, retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        mode_text = "Hover"

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            fingers = fingers_up(hand_landmarks, handedness_label)
            thumb, index, middle, ring, pinky = fingers

            # fingertip pixel coordinates
            index_tip = hand_landmarks.landmark[8]
            ix, iy = int(index_tip.x * CAM_WIDTH), int(index_tip.y * CAM_HEIGHT)

            # --- smoothing so the line isn't jittery ---
            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = ix, iy
            smooth_x = smooth_x + (ix - smooth_x) // SMOOTHING
            smooth_y = smooth_y + (iy - smooth_y) // SMOOTHING

            if index and middle and not ring and not pinky:
                # --- SELECTION MODE ---
                mode_text = "Selection"
                prev_x, prev_y = 0, 0  # lift the pen
                cv2.circle(frame, (smooth_x, smooth_y), 15, selected_color, cv2.FILLED)

                if smooth_y < SWATCH_H:
                    idx = min(smooth_x // SWATCH_W, len(PALETTE) - 1)
                    selected_name, selected_color = PALETTE[idx]

            elif index and not middle and not ring and not pinky:
                # --- DRAW MODE ---
                mode_text = f"Drawing ({selected_name})"
                cv2.circle(frame, (smooth_x, smooth_y), 10, selected_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = smooth_x, smooth_y

                thickness = ERASER_THICKNESS if selected_name == "Eraser" else BRUSH_THICKNESS
                draw_color = (0, 0, 0) if selected_name == "Eraser" else selected_color

                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, thickness)
                prev_x, prev_y = smooth_x, smooth_y

            else:
                # any other gesture (fist, all-open, etc.) = pause
                mode_text = "Paused"
                prev_x, prev_y = 0, 0
        else:
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0

        # --- merge canvas onto live frame ---
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame_bg = cv2.bitwise_and(frame, mask_3ch)
        output = cv2.bitwise_or(frame_bg, canvas)

        # --- UI overlays ---
        draw_palette(output, selected_name)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(output, f"FPS: {int(fps)}", (CAM_WIDTH - 150, CAM_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, mode_text, (20, CAM_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, "c=clear  s=save  q=quit", (20, CAM_HEIGHT - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Draw Without Touching", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
        elif key == ord('s'):
            filename = f"drawings/drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved -> {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
