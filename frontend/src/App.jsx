// App.jsx
// Root component. Owns the webcam feed, the MediaPipe detection loop,
// and the letter display. All rendering happens here; the utility modules
// (mediapipe.js, modelInference.js, heuristic.js) contain no UI code.

import { useState, useEffect, useRef } from 'react';
import { initMediaPipe, detectHands } from './utils/mediapipe';
import { loadModel, isModelLoaded, runModelInference } from './utils/modelInference';
import { recognizeASL } from './utils/heuristic';
import './App.css';

// MediaPipe landmark connectivity — pairs of indices that should be connected
// by a line when drawing the hand skeleton on the canvas overlay.
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [5,9],[9,10],[10,11],[11,12],  // middle
  [9,13],[13,14],[14,15],[15,16],// ring
  [13,17],[17,18],[18,19],[19,20],// pinky
  [0,17],                        // palm base
];

export default function App() {
  // Which recognition mode is active. 'heuristic' uses rule-based logic;
  // 'model' uses the TF.js CNN (only available if convert_model.py has been run).
  const [mode, setMode] = useState('heuristic');
  // The currently recognized ASL letter (or '' when no hand is visible).
  const [letter, setLetter] = useState('');
  // Status text shown while the app is loading (mediapipe, camera, model).
  const [status, setStatus] = useState('Initializing...');
  // True once the camera stream is running and the detection loop has started.
  const [ready, setReady] = useState(false);
  // True only if loadModel() succeeded — gates the ML Model button.
  const [modelAvailable, setModelAvailable] = useState(false);

  const videoRef = useRef(null);   // <video> element that receives the webcam stream
  const canvasRef = useRef(null);  // <canvas> overlay drawn on top of the video
  // modeRef mirrors the mode state so the RAF loop can read it without stale closures.
  const modeRef = useRef('heuristic');
  const rafRef = useRef(null);          // stores the requestAnimationFrame handle for cleanup
  const initializedRef = useRef(false); // prevents the effect from running twice in React StrictMode

  // Keep modeRef in sync whenever the mode state changes.
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // One-time setup effect: load MediaPipe + model, start camera, then start the loop.
  useEffect(() => {
    // active flag lets the cleanup function stop the RAF loop when the component unmounts.
    let active = true;

    async function init() {
      // initializedRef prevents this block from running a second time in React StrictMode,
      // which intentionally mounts/unmounts components twice in development.
      if (!initializedRef.current) {
        initializedRef.current = true;

        // Step 1: download and initialize MediaPipe WASM + hand model from CDN.
        setStatus('Loading MediaPipe...');
        try {
          await initMediaPipe();
          console.log('MediaPipe initialized successfully.');
        } catch (e) {
          console.error('MediaPipe failed to initialize:', e);
          setStatus('MediaPipe failed to load. Check the console for details.');
          return; // nothing will work without MediaPipe, so stop here
        }

        // Step 2: try to load the TF.js model from /public/model/.
        // This is optional — if the files don't exist the app still works in heuristic mode.
        setStatus('Loading ML model...');
        try {
          await loadModel();
          setModelAvailable(true);
        } catch (e) {
          // Log so the developer can see the real error (e.g. bad model conversion).
          console.error('loadModel failed:', e);
        }

        // Step 3: request webcam access and attach the stream to the <video> element.
        setStatus('Starting camera...');
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = videoRef.current;
        video.srcObject = stream;
        // Wait for the browser to know the video dimensions before playing.
        await new Promise((res) => { video.onloadedmetadata = res; });
        video.play();

        setReady(true);
        setStatus('');
      }

      // Start the detection/render loop now that everything is initialized.
      if (active) startLoop(active);
    }

    init();

    // Cleanup: cancel the RAF loop when the component unmounts.
    return () => {
      active = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Draw the 21-point hand skeleton on the canvas.
  // landmarks: array of {x,y,z} in normalized [0,1] coords.
  // w, h: canvas intrinsic pixel dimensions (= video resolution).
  function drawLandmarks(ctx, landmarks, w, h) {
    // Draw the bone connections first (under the dots).
    ctx.strokeStyle = '#00ff41';
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      ctx.stroke();
    }
    // Draw a dot at each of the 21 landmark positions.
    ctx.fillStyle = '#00aaff';
    for (const lm of landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Starts the requestAnimationFrame render loop.
  // active is a reference to the closure variable in the effect; when it becomes
  // false (on unmount) the loop stops without needing cancelAnimationFrame.
  function startLoop(active) {
    // inferring flag prevents multiple async inference calls from queuing up.
    // (Not actually async here, but kept as a guard for future changes.)
    let inferring = false;

    function loop() {
      if (!active) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      // Match the canvas's internal resolution to the camera resolution each frame.
      // Setting canvas.width/height also clears the canvas, so no explicit clearRect needed,
      // but we still call clearRect below for clarity.
      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      canvas.width = w;
      canvas.height = h;

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, w, h);

      // Only run detection once the video has enough data to display a frame.
      // readyState >= 2 (HAVE_CURRENT_DATA) means at least one frame is available.
      if (video.readyState >= 2) {
        const result = detectHands(video);

        if (result?.handLandmarks?.length > 0) {
          // Take the first detected hand (numHands: 1, so there's only ever one).
          const landmarks = result.handLandmarks[0];
          drawLandmarks(ctx, landmarks, w, h);

          if (!inferring) {
            inferring = true;
            const currentMode = modeRef.current; // read from ref to avoid stale closure

            if (currentMode === 'heuristic') {
              // Rule-based recognizer — synchronous, no model needed.
              setLetter(recognizeASL(landmarks));
              inferring = false;
            } else if (isModelLoaded()) {
              // CNN inference — synchronous because tf.tidy returns dataSync().
              setLetter(runModelInference(video, landmarks));
              inferring = false;
            } else {
              inferring = false;
            }
          }
        } else {
          // No hand detected — clear the letter display.
          setLetter('');
        }
      }

      // Schedule the next frame.
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  return (
    <div className="app">
      <header className="top-bar">
        <span className="title">ASL Translator</span>
        <div className="toggle-group">
          {/* Heuristic mode is always available */}
          <button
            className={mode === 'heuristic' ? 'active' : ''}
            onClick={() => setMode('heuristic')}
          >
            Heuristic
          </button>
          {/* ML Model button is disabled until loadModel() succeeds.
              To enable: run convert_model.py to generate /public/model/ files. */}
          <button
            className={mode === 'model' ? 'active' : ''}
            onClick={() => setMode('model')}
            disabled={!modelAvailable}
            title={modelAvailable ? '' : 'Run convert_model.py first'}
          >
            ML Model
          </button>
        </div>
      </header>

      <div className="camera-wrap">
        {/* The video element receives the raw webcam stream — never drawn to directly. */}
        <video ref={videoRef} className="video" playsInline muted />
        {/* The canvas sits on top of the video (via CSS position:absolute) and is where
            the skeleton overlay is drawn. pointer-events:none lets clicks pass through. */}
        <canvas ref={canvasRef} className="canvas" />

        {/* Loading indicator shown while status is non-empty (before camera starts). */}
        {!ready && (
          <div className="status-overlay">{status}</div>
        )}

        {/* Large letter badge shown at the bottom of the frame when a letter is recognized. */}
        {letter && (
          <div className="letter-badge">{letter}</div>
        )}
      </div>
    </div>
  );
}
