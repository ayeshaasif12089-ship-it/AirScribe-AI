import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


st.set_page_config(
    page_title="AirScribe AI",
    page_icon="✋",
    layout="wide"
)

st.title("✋ AirScribe AI")
st.markdown(
    "### Gesture-Controlled Virtual Painter"
)
st.write(
    "Use your webcam and hand gestures to draw in the air using real-time computer vision."
)


class AirScribeProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()

        self.canvas = None
        self.prev_point = None

        self.selected_color = (0, 0, 255)
        self.selected_name = "Red"

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.colors = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255),
            "Purple": (255, 0, 255),
            "Eraser": (0, 0, 0)
        }

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        height, width, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        mode = "No Hand Detected"

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = hand.landmark

            index_tip = landmarks[
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            middle_tip = landmarks[
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]

            index_pip = landmarks[
                self.mp_hands.HandLandmark.INDEX_FINGER_PIP
            ]

            middle_pip = landmarks[
                self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ]

            x = int(index_tip.x * width)
            y = int(index_tip.y * height)

            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            # Draw when only index finger is up
            if index_up and not middle_up:

                mode = "DRAWING"

                if self.prev_point is not None:

                    cv2.line(
                        self.canvas,
                        self.prev_point,
                        (x, y),
                        self.selected_color,
                        6
                    )

                self.prev_point = (x, y)

            # Selection mode
            elif index_up and middle_up:

                mode = "COLOR SELECT"

                self.prev_point = None

                # Color palette at top
                palette = [
                    ("Red", 50),
                    ("Green", 120),
                    ("Blue", 190),
                    ("Yellow", 260),
                    ("Purple", 330),
                    ("Eraser", 400)
                ]

                for name, px in palette:

                    if px - 25 < x < px + 25 and 20 < y < 70:

                        self.selected_name = name
                        self.selected_color = self.colors[name]

            # Pause
            else:

                mode = "PAUSED"
                self.prev_point = None

        else:

            self.prev_point = None

        # Merge drawing canvas with webcam
        output = cv2.addWeighted(
            img,
            0.75,
            self.canvas,
            0.25,
            0
        )

        # Palette
        palette = [
            ("Red", (0, 0, 255), 50),
            ("Green", (0, 255, 0), 120),
            ("Blue", (255, 0, 0), 190),
            ("Yellow", (0, 255, 255), 260),
            ("Purple", (255, 0, 255), 330),
            ("Eraser", (255, 255, 255), 400)
        ]

        for name, color, px in palette:

            cv2.circle(
                output,
                (px, 45),
                22,
                color,
                -1
            )

            if name == self.selected_name:

                cv2.circle(
                    output,
                    (px, 45),
                    28,
                    (255, 255, 255),
                    2
                )

        cv2.putText(
            output,
            f"Mode: {mode}",
            (20, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Color: {self.selected_name}",
            (20, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )

    def clear_canvas(self):

        with self.lock:

            if self.canvas is not None:

                self.canvas[:] = 0


st.markdown("### 🎨 Controls")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "☝️ **Index finger only** → Draw"
    )

with col2:
    st.info(
        "✌️ **Index + middle fingers** → Select color"
    )

with col3:
    st.info(
        "✊ **Other gestures** → Pause"
    )


ctx = webrtc_streamer(
    key="airscribe",
    video_processor_factory=AirScribeProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration=RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                }
            ]
        }
    ),
)


if ctx.video_processor:

    if st.button("🧹 Clear Canvas"):

        ctx.video_processor.clear_canvas()

        st.rerun()


st.markdown("---")

st.caption(
    "AirScribe AI • Computer Vision • MediaPipe • OpenCV • Streamlit"
)
