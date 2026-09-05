import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";


// =========================================
// ELEMENTS
// =========================================

const video =
    document.getElementById("video");

const drawCanvas =
    document.getElementById("drawCanvas");

const landmarkCanvas =
    document.getElementById("landmarkCanvas");

const stage =
    document.getElementById("stage");

const drawCtx =
    drawCanvas.getContext("2d");

const landmarkCtx =
    landmarkCanvas.getContext("2d");

const startBtn =
    document.getElementById("startBtn");

const clearBtn =
    document.getElementById("clearBtn");

const saveBtn =
    document.getElementById("saveBtn");

const cameraPlaceholder =
    document.getElementById(
        "cameraPlaceholder"
    );

const loading =
    document.getElementById("loading");

const systemStatus =
    document.getElementById("systemStatus");

const systemDot =
    document.getElementById("systemDot");

const modeText =
    document.getElementById("modeText");

const modeDot =
    document.getElementById("modeDot");

const gestureText =
    document.getElementById("gestureText");

const selectedColorText =
    document.getElementById("selectedColor");

const brushSize =
    document.getElementById("brushSize");

const brushValue =
    document.getElementById("brushValue");

const aiStatus =
    document.getElementById("aiStatus");

const handCount =
    document.getElementById("handCount");


// =========================================
// STATE
// =========================================

let handLandmarker = null;

let stream = null;

let running = false;

let lastVideoTime = -1;

let selectedColor = "#ff4d67";

let selectedColorName = "Red";

let brushWidth = 6;

let previousPoint = null;

let lastTimestamp = 0;

let lastSelectionTime = 0;


// =========================================
// MEDIAPIPE MODEL
// =========================================

async function createHandLandmarker() {

    try {

        systemStatus.textContent =
            "Loading AI model...";

        aiStatus.textContent =
            "Loading";

        const vision =
            await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            );


        handLandmarker =
            await HandLandmarker.createFromOptions(
                vision,
                {
                    baseOptions: {

                        modelAssetPath:
                            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

                    },

                    runningMode: "VIDEO",

                    numHands: 1,

                    minHandDetectionConfidence: 0.5,

                    minHandPresenceConfidence: 0.5,

                    minTrackingConfidence: 0.5
                }
            );


        systemStatus.textContent =
            "AI Ready";

        systemDot.style.background =
            "#35d07f";

        aiStatus.textContent =
            "Ready";

    } catch (error) {

        console.error(error);

        systemStatus.textContent =
            "AI failed to load";

        systemDot.style.background =
            "#ff4d67";

        aiStatus.textContent =
            "Error";

        alert(
            "MediaPipe could not load. Please refresh the page and try again."
        );
    }
}


// =========================================
// CAMERA
// =========================================

async function startCamera() {

    if (!handLandmarker) {

        alert(
            "AI is still loading. Please wait a moment."
        );

        return;
    }


    try {

        loading.classList.remove("hidden");

        stream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        facingMode: "user",
                        width: {
                            ideal: 1280
                        },
                        height: {
                            ideal: 720
                        }
                    },

                    audio: false
                }
            );


        video.srcObject = stream;

        video.style.display = "block";

        cameraPlaceholder.style.display =
            "none";

        running = true;

        startBtn.innerHTML =
            "<span>⏹</span> Stop Camera";

        modeText.textContent =
            "Live";

        modeDot.style.background =
            "#35d07f";

        loading.classList.add("hidden");

        await video.play();

        resizeCanvases();

        requestAnimationFrame(
            predictWebcam
        );

    } catch (error) {

        console.error(error);

        loading.classList.add("hidden");

        alert(
            "Camera access was blocked. Please allow camera access in your browser and try again."
        );
    }
}


// =========================================
// STOP CAMERA
// =========================================

function stopCamera() {

    running = false;

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        stream = null;
    }

    video.srcObject = null;

    video.style.display = "none";

    cameraPlaceholder.style.display =
        "flex";

    startBtn.innerHTML =
        "<span>📷</span> Start Camera";

    modeText.textContent =
        "Camera Off";

    modeDot.style.background =
        "#777";

    gestureText.textContent =
        "Waiting for hand...";

    handCount.textContent =
        "0";

    previousPoint = null;

    clearLandmarks();
}


// =========================================
// TOGGLE CAMERA
// =========================================

startBtn.addEventListener(
    "click",
    async () => {

        if (running) {

            stopCamera();

        } else {

            await startCamera();

        }

    }
);


// =========================================
// CANVAS SIZE
// =========================================

function resizeCanvases() {

    const width =
        video.videoWidth ||
        stage.clientWidth;

    const height =
        video.videoHeight ||
        stage.clientHeight;


    drawCanvas.width = width;

    drawCanvas.height = height;

    landmarkCanvas.width = width;

    landmarkCanvas.height = height;
}


// =========================================
// WINDOW RESIZE
// =========================================

window.addEventListener(
    "resize",
    () => {

        if (running) {

            resizeCanvases();

        }

    }
);


// =========================================
// PREDICT WEBCAM
// =========================================

async function predictWebcam() {

    if (!running) {
        return;
    }


    if (
        video.readyState <
        2
    ) {

        requestAnimationFrame(
            predictWebcam
        );

        return;
    }


    const now =
        performance.now();


    if (
        video.currentTime !==
        lastVideoTime
    ) {

        lastVideoTime =
            video.currentTime;


        try {

            const results =
                handLandmarker.detectForVideo(
                    video,
                    now
                );

            processResults(
                results
            );

        } catch (error) {

            console.error(
                "Detection error:",
                error
            );
        }
    }


    requestAnimationFrame(
        predictWebcam
    );
}


// =========================================
// PROCESS RESULTS
// =========================================

function processResults(
    results
) {

    clearLandmarks();


    const hands =
        results.landmarks || [];


    handCount.textContent =
        hands.length;


    if (!hands.length) {

        gestureText.textContent =
            "Waiting for hand...";

        modeText.textContent =
            "No Hand";

        previousPoint = null;

        return;
    }


    const landmarks =
        hands[0];


    drawLandmarks(
        landmarks
    );


    const gesture =
        detectGesture(
            landmarks
        );


    gestureText.textContent =
        gesture.label;


    modeText.textContent =
        gesture.mode;


    // -------------------------
    // DRAW
    // -------------------------

    if (
        gesture.type ===
        "draw"
    ) {

        const point =
            landmarks[
                8
            ];

        const x =
            point.x *
            drawCanvas.width;

        const y =
            point.y *
            drawCanvas.height;


        drawStroke(
            x,
            y
        );

    } else {

        previousPoint = null;

    }


    // -------------------------
    // COLOR SELECT
    // -------------------------

    if (
        gesture.type ===
        "select"
    ) {

        const point =
            landmarks[8];

        selectColorFromGesture(
            point.x,
            point.y
        );

    }
}


// =========================================
// GESTURE DETECTION
// =========================================

function detectGesture(
    landmarks
) {

    const indexUp =
        isFingerUp(
            landmarks,
            8,
            6
        );


    const middleUp =
        isFingerUp(
            landmarks,
            12,
            10
        );


    const ringUp =
        isFingerUp(
            landmarks,
            16,
            14
        );


    const pinkyUp =
        isFingerUp(
            landmarks,
            20,
            18
        );


    const thumbUp =
        isThumbUp(
            landmarks
        );


    const fingersUp =
        [
            thumbUp,
            indexUp,
            middleUp,
            ringUp,
            pinkyUp
        ]
        .filter(Boolean)
        .length;


    // Index only
    if (
        indexUp &&
        !middleUp &&
        !ringUp &&
        !pinkyUp
    ) {

        return {

            type: "draw",

            mode: "Drawing",

            label: "☝️ Drawing"

        };
    }


    // Index + middle
    if (
        indexUp &&
        middleUp &&
        !ringUp &&
        !pinkyUp
    ) {

        return {

            type: "select",

            mode: "Color Select",

            label: "✌️ Selecting Color"

        };
    }


    // Open hand
    if (
        fingersUp >= 4
    ) {

        return {

            type: "pause",

            mode: "Paused",

            label: "🖐️ Paused"

        };
    }


    // Fist
    if (
        fingersUp === 0
    ) {

        return {

            type: "pause",

            mode: "Paused",

            label: "✊ Paused"

        };
    }


    return {

        type: "pause",

        mode: "Paused",

        label: "✋ Ready"

    };
}


// =========================================
// FINGER DETECTION
// =========================================

function isFingerUp(
    landmarks,
    tip,
    pip
) {

    return (
        landmarks[tip].y <
        landmarks[pip].y
    );
}


function isThumbUp(
    landmarks
) {

    return (
        landmarks[4].x >
        landmarks[3].x
    );
}


// =========================================
// DRAW
// =========================================

function drawStroke(
    x,
    y
) {

    if (!previousPoint) {

        previousPoint = {
            x,
            y
        };

        return;
    }


    drawCtx.save();

    drawCtx.lineWidth =
        brushWidth;

    drawCtx.lineCap =
        "round";

    drawCtx.lineJoin =
        "round";


    if (
        selectedColorName ===
        "Eraser"
    ) {

        drawCtx.globalCompositeOperation =
            "destination-out";

        drawCtx.strokeStyle =
            "rgba(0,0,0,1)";

    } else {

        drawCtx.globalCompositeOperation =
            "source-over";

        drawCtx.strokeStyle =
            selectedColor;
    }


    drawCtx.beginPath();

    drawCtx.moveTo(
        previousPoint.x,
        previousPoint.y
    );

    drawCtx.lineTo(
        x,
        y
    );

    drawCtx.stroke();

    drawCtx.restore();


    previousPoint = {
        x,
        y
    };
}


// =========================================
// LANDMARKS
// =========================================

const connections = [

    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],

    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],

    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],

    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],

    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],

    [0, 17]
];


function drawLandmarks(
    landmarks
) {

    landmarkCtx.save();

    landmarkCtx.lineWidth =
        2;

    landmarkCtx.strokeStyle =
        "rgba(108,99,255,0.8)";

    landmarkCtx.fillStyle =
        "#ffffff";


    for (
        const [
            start,
            end
        ]
        of connections
    ) {

        const a =
            landmarks[start];

        const b =
            landmarks[end];


        landmarkCtx.beginPath();

        landmarkCtx.moveTo(
            a.x *
                landmarkCanvas.width,
            a.y *
                landmarkCanvas.height
        );

        landmarkCtx.lineTo(
            b.x *
                landmarkCanvas.width,
            b.y *
                landmarkCanvas.height
        );

        landmarkCtx.stroke();
    }


    for (
        const point
        of landmarks
    ) {

        landmarkCtx.beginPath();

        landmarkCtx.arc(
            point.x *
                landmarkCanvas.width,
            point.y *
                landmarkCanvas.height,
            4,
            0,
            Math.PI * 2
        );

        landmarkCtx.fill();
    }


    landmarkCtx.restore();
}


// =========================================
// CLEAR LANDMARKS
// =========================================

function clearLandmarks() {

    landmarkCtx.clearRect(
        0,
        0,
        landmarkCanvas.width,
        landmarkCanvas.height
    );
}


// =========================================
// COLOR BUTTONS
// =========================================

document
    .querySelectorAll(
        ".color-btn"
    )
    .forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    setColor(
                        button.dataset.color,
                        button.dataset.name
                    );

                }
            );

        }
    );


function setColor(
    color,
    name
) {

    selectedColor =
        color;

    selectedColorName =
        name;

    selectedColorText.textContent =
        name;


    document
        .querySelectorAll(
            ".color-btn"
        )
        .forEach(
            btn =>
                btn.classList.remove(
                    "active"
                )
        );


    const selected =
        document.querySelector(
            `[data-name="${name}"]`
        );


    if (selected) {

        selected.classList.add(
            "active"
        );
    }
}


// =========================================
// GESTURE COLOR SELECTION
// =========================================

function selectColorFromGesture(
    normalizedX,
    normalizedY
) {

    const now =
        performance.now();


    if (
        now -
        lastSelectionTime <
        500
    ) {

        return;
    }


    // Palette is near top
    if (
        normalizedY >
        0.16
    ) {

        return;
    }


    const colors = [

        {
            name: "Red",
            min: 0.05,
            max: 0.18
        },

        {
            name: "Green",
            min: 0.19,
            max: 0.32
        },

        {
            name: "Blue",
            min: 0.33,
            max: 0.46
        },

        {
            name: "Yellow",
            min: 0.47,
            max: 0.60
        },

        {
            name: "Purple",
            min: 0.61,
            max: 0.74
        },

        {
            name: "Eraser",
            min: 0.75,
            max: 0.90
        }

    ];


    const selected =
        colors.find(
            item =>
                normalizedX >= item.min &&
                normalizedX <= item.max
        );


    if (!selected) {

        return;
    }


    const button =
        document.querySelector(
            `[data-name="${selected.name}"]`
        );


    if (!button) {

        return;
    }


    setColor(
        button.dataset.color,
        button.dataset.name
    );


    lastSelectionTime =
        now;
}


// =========================================
// CLEAR CANVAS
// =========================================

clearBtn.addEventListener(
    "click",
    () => {

        drawCtx.clearRect(
            0,
            0,
            drawCanvas.width,
            drawCanvas.height
        );

        previousPoint = null;

    }
);


// =========================================
// SAVE DRAWING
// =========================================

saveBtn.addEventListener(
    "click",
    () => {

        const link =
            document.createElement(
                "a"
            );


        link.download =
            "airscribe-drawing.png";


        link.href =
            drawCanvas.toDataURL(
                "image/png"
            );


        link.click();

    }
);


// =========================================
// BRUSH SIZE
// =========================================

brushSize.addEventListener(
    "input",
    () => {

        brushWidth =
            Number(
                brushSize.value
            );

        brushValue.textContent =
            `${brushWidth}px`;

    }
);


// =========================================
// INITIALIZE
// =========================================

createHandLandmarker();
