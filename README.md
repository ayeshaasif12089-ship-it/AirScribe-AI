AirScribe-AI ✍️🖐️

Draw in the air — no mouse, no touchscreen, no stylus.
AirScribe-AI uses your webcam and real-time hand tracking to turn your index finger into a paintbrush. Built with OpenCV and MediaPipe.
✨ Features


Real-time hand & finger tracking using MediaPipe's 21-point hand landmark model
Gesture-based controls — no keyboard needed while drawing
Color palette — Red, Green, Blue, Yellow, Purple, and an Eraser, selectable in-air
Smoothing to reduce jitter and produce clean lines
Save your art as a PNG with a single keypress
Runs fully locally — no cloud API calls, no internet required after setup


🖐️ Gesture Controls

GestureAction☝️ Index finger onlyDraw mode✌️ Index + middle fingerSelection mode (move freely, pick a color)✋ All fingers upPaused👊 FistPaused

KeyActioncClear canvassSave drawing as PNGqQuit

🛠️ Tech Stack


Python 3.10+
OpenCV — video capture, image processing, rendering
MediaPipe — hand landmark detection
NumPy — canvas array operations


🚀 Getting Started

1. Clone the repo

bashgit clone https://github.com/<your-username>/AirScribe-AI.git
cd AirScribe-AI

2. Create a virtual environment

bashpython -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

3. Install dependencies

bashpip install -r requirements.txt

4. Run it

bashpython main.py

A webcam window will open. Hold up your index finger and start drawing!

🧩 How It Works


Capture — OpenCV grabs frames from the webcam.
Detect — MediaPipe processes each frame and returns 21 hand landmarks.
Interpret — the app checks which fingers are extended to decide the current mode (draw / select / pause).
Track — the index fingertip (landmark 8) position is smoothed frame-to-frame to reduce jitter.
Render — strokes are drawn onto a persistent canvas, which is composited over the live video feed each frame.


🗺️ Roadmap


 Shape recognition (auto-clean circles, lines, rectangles)
 Undo/redo via gesture
 Multi-hand support (two-hand drawing or gesture shortcuts)
 Adjustable brush size via pinch gesture
 Export drawing as SVG in addition to PNG


📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

🙌 Acknowledgements


Google MediaPipe for the hand-tracking model
OpenCV community