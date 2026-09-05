import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AirScribe AI",
    page_icon="✋",
    layout="wide"
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
    .hero {
        padding: 2rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.15rem;
        color: #aeb6c4;
    }

    .card {
        padding: 1.2rem;
        border-radius: 14px;
        background: #171a21;
        border: 1px solid #2b303b;
        margin-bottom: 1rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #202633;
        margin-right: 0.4rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>✋ AirScribe AI</h1>
    <p>
        Gesture-inspired virtual painting powered by
        Computer Vision and MediaPipe.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<span class="badge">Computer Vision</span>
<span class="badge">MediaPipe</span>
<span class="badge">OpenCV</span>
<span class="badge">Interactive Drawing</span>
""", unsafe_allow_html=True)

st.divider()


# -----------------------------
# Feature cards
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>☝️ Gesture Drawing</h3>
        <p>AirScribe interprets hand landmarks to enable gesture-controlled drawing.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🖐️ Hand Tracking</h3>
        <p>MediaPipe detects and tracks 21 hand landmarks from an image.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>🎨 Digital Canvas</h3>
        <p>Experiment with colors and create drawings directly in the browser.</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Hand landmark detector
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def detect_hand(image):
    """
    Detect hand landmarks and return an annotated image.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(image_rgb)

    annotated = image.copy()

    hand_count = 0

    if results.multi_hand_landmarks:

        hand_count = len(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    return annotated, hand_count


# -----------------------------
# Hand tracking section
# -----------------------------
st.header("🖐️ Computer Vision Demo")

st.write(
    "Upload a hand image and AirScribe will detect the hand landmarks "
    "using MediaPipe."
)

uploaded_file = st.file_uploader(
    "Upload a hand image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    annotated, hand_count = detect_hand(image_bgr)

    annotated_rgb = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Original")

        st.image(
            image,
            use_container_width=True
        )

    with right:

        st.subheader("AirScribe Analysis")

        st.image(
            annotated_rgb,
            use_container_width=True
        )

    if hand_count > 0:

        st.success(
            f"Detected {hand_count} hand(s) and extracted hand landmarks."
        )

    else:

        st.warning(
            "No hand detected. Try uploading a clearer image with the hand visible."
        )


st.divider()


# -----------------------------
# Interactive canvas
# -----------------------------
st.header("🎨 Interactive AirScribe Canvas")

st.write(
    "Use the canvas below to simulate the digital painting experience."
)


col1, col2 = st.columns([1, 3])

with col1:

    drawing_mode = st.selectbox(
        "Drawing mode",
        [
            "freedraw",
            "line",
            "rect",
            "circle"
        ]
    )

    stroke_color = st.color_picker(
        "Brush color",
        "#4F8CFF"
    )

    stroke_width = st.slider(
        "Brush size",
        1,
        30,
        6
    )

with col2:

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#0f1117",
        height=450,
        width=750,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="airscribe_canvas"
    )


# -----------------------------
# Canvas information
# -----------------------------
if canvas_result.image_data is not None:

    st.caption(
        "Your drawing is being captured interactively by the AirScribe canvas."
    )


st.divider()


# -----------------------------
# Gesture logic explanation
# -----------------------------
st.header("🧠 How AirScribe Works")

step1, step2, step3 = st.columns(3)

with step1:

    st.markdown("""
    ### 01 — Detect
    MediaPipe identifies the hand and its landmark points.
    """)

with step2:

    st.markdown("""
    ### 02 — Interpret
    Finger positions can be interpreted as drawing or selection gestures.
    """)

with step3:

    st.markdown("""
    ### 03 — Create
    The gesture is translated into a digital drawing interaction.
    """)


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "AirScribe AI • Built with Python, OpenCV, MediaPipe and Streamlit"
)
