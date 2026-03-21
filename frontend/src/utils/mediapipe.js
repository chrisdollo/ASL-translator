// mediapipe.js
// Wraps the MediaPipe Tasks-Vision HandLandmarker.
// Two exports are used by App.jsx:
//   initMediaPipe() — downloads the WASM runtime + hand model from CDN, must be awaited once on startup
//   detectHands(videoElement) — runs detection on the current video frame, returns landmark data

import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Holds the single HandLandmarker instance after init. Null until initMediaPipe() completes.
let handLandmarker = null;

export async function initMediaPipe() {
  // FilesetResolver fetches the MediaPipe WASM binary from jsDelivr CDN.
  // This is the low-level WebAssembly runtime that runs the ML graph in the browser.
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  // Create the HandLandmarker. This also downloads the hand_landmarker.task model file
  // (~8 MB) from Google's CDN on first load (cached by the browser afterwards).
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      // No delegate specified — defaults to CPU via XNNPACK, which works on all devices.
      // GPU delegate can be faster but fails silently on some macOS/Safari setups.
    },
    runningMode: 'VIDEO',   // VIDEO mode keeps a Kalman filter for tracking across frames (smoother than IMAGE mode)
    numHands: 1,            // only track one hand at a time
    minHandDetectionConfidence: 0.6,  // how confident the model must be to initially detect a hand
    minHandPresenceConfidence: 0.5,   // how confident it must be that the hand is still present between frames
    minTrackingConfidence: 0.5,       // how confident the tracker must be to keep tracking without re-detecting
  });
}

export function detectHands(videoElement) {
  // Guard: if initMediaPipe() hasn't finished yet, return null so the render loop skips drawing.
  if (!handLandmarker) return null;

  // detectForVideo needs a monotonically increasing timestamp in milliseconds.
  // performance.now() is the right choice for webcam streams (no fixed frame rate).
  // Returns an object with .handLandmarks: array of hands, each hand = array of 21 {x,y,z} points.
  return handLandmarker.detectForVideo(videoElement, performance.now());
}
