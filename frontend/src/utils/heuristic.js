// heuristic.js
// Rule-based ASL letter recognizer using MediaPipe hand landmarks.
// Ported from ASLHeuristicIdentifier.ipynb — logic must stay in sync with the notebook.
//
// MediaPipe landmark numbering (all coords normalized 0-1, origin = top-left):
//   0  = wrist
//   1-4  = thumb  (CMC, MCP, IP, tip)
//   5-8  = index  (MCP, PIP, DIP, tip)
//   9-12 = middle (MCP, PIP, DIP, tip)
//   13-16= ring   (MCP, PIP, DIP, tip)
//   17-20= pinky  (MCP, PIP, DIP, tip)
//
// Because y increases downward, a fingertip with a SMALLER y value than its knuckle
// means the finger is pointing UP (extended). This is the basis for fingerState().

// Euclidean distance between two landmarks in normalized 2D (x,y) space.
function dist(a, b) {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
}

// Classify a single finger as OPEN (extended), CLOSE (curled), or HALF_OPEN (bent).
// mcp, pip, tip are landmark indices for that finger's knuckle, middle joint, and tip.
function fingerState(lm, mcp, pip, tip) {
  // OPEN: tip is above pip, pip is above mcp (finger pointing up)
  if (lm[mcp].y > lm[pip].y && lm[pip].y > lm[tip].y) return 'OPEN';
  // CLOSE: tip is below pip, pip is below mcp (finger fully curled down)
  if (lm[mcp].y < lm[pip].y && lm[pip].y < lm[tip].y) return 'CLOSE';
  // HALF_OPEN: mcp above pip but tip curled back above pip (finger bent at middle)
  if (lm[mcp].y > lm[pip].y && lm[tip].y > lm[pip].y) return 'HALF_OPEN';
  return '';
}

// Main entry point — takes 21 landmarks and returns the recognized ASL letter, or ''.
export function recognizeASL(lm) {
  if (!lm || lm.length < 21) return '';

  // Thumb uses x-axis (left/right) instead of y-axis because it extends sideways.
  // For a right hand: tip moves left (smaller x) when thumb is extended.
  let thumbState = '';
  if (lm[2].x > lm[3].x && lm[3].x > lm[4].x) thumbState = 'CLOSE';
  else if (lm[2].x < lm[3].x && lm[3].x < lm[4].x) thumbState = 'OPEN';

  // Classify the four fingers using the y-axis knuckle/tip comparison.
  // Landmark pairs: (MCP=5,PIP=6,tip=8) for index, etc. — see header for numbering.
  const indexState  = fingerState(lm, 6,  7,  8);
  const middleState = fingerState(lm, 10, 11, 12);
  const ringState   = fingerState(lm, 14, 15, 16);
  const pinkyState  = fingerState(lm, 18, 19, 20);

  // ref = distance from wrist(0) to index MCP(1), used as a scale reference so
  // distance thresholds work regardless of how close the hand is to the camera.
  const ref   = dist(lm[0], lm[1]);
  // Key pairwise distances used by several letter rules below.
  const d_t_i = dist(lm[4], lm[8]);   // thumb tip to index tip
  const d_t_m = dist(lm[4], lm[12]);  // thumb tip to middle tip
  const d_i_m = dist(lm[8], lm[12]);  // index tip to middle tip

  // True when the finger is pointing more horizontally than vertically.
  const index_horiz  = Math.abs(lm[8].x  - lm[6].x)  > Math.abs(lm[8].y  - lm[6].y);
  const middle_horiz = Math.abs(lm[12].x - lm[10].x) > Math.abs(lm[12].y - lm[10].y);

  // ── Letter classification ────────────────────────────────────────────────────
  // Rules mirror the Python notebook logic exactly. Order matters: more specific
  // checks come after broad ones so they can override.

  // B: all four fingers extended, thumb in any state
  if (indexState === 'OPEN' && middleState === 'OPEN' && ringState === 'OPEN' && pinkyState === 'OPEN')
    return 'B';

  // W: index, middle, ring extended — pinky curled
  if (indexState === 'OPEN' && middleState === 'OPEN' && ringState === 'OPEN' && pinkyState === 'CLOSE')
    return 'W';

  // Y: thumb and pinky out, three middle fingers curled
  if (thumbState === 'OPEN' && indexState === 'CLOSE' && middleState === 'CLOSE' && ringState === 'CLOSE' && pinkyState === 'OPEN')
    return 'Y';

  // L: thumb and index extended, rest curled
  if (thumbState === 'OPEN' && indexState === 'OPEN' && middleState === 'CLOSE' && ringState === 'CLOSE' && pinkyState === 'CLOSE')
    return 'L';

  // I: only pinky extended (index/middle/ring curled)
  if (indexState === 'CLOSE' && middleState === 'CLOSE' && ringState === 'CLOSE' && pinkyState === 'OPEN')
    return 'I';

  // Index + middle extended, ring + pinky curled — could be H, K, V, R, or U.
  // Disambiguate by orientation and thumb/finger distances.
  if (indexState === 'OPEN' && middleState === 'OPEN' && ringState === 'CLOSE' && pinkyState === 'CLOSE') {
    if (index_horiz && middle_horiz) return 'H';               // both horizontal → H
    if (thumbState === 'OPEN' && d_t_m < 1.5 * ref) return 'K'; // thumb near middle → K
    if (d_i_m > 1.5 * ref) return 'V';                          // fingers spread wide → V
    if (Math.abs(lm[8].x - lm[12].x) < 0.4 * ref) return 'R';  // tips very close (crossed) → R
    return 'U';                                                  // default: fingers together → U
  }

  // Only index extended, rest curled — could be G or D.
  if (indexState === 'OPEN' && middleState === 'CLOSE' && ringState === 'CLOSE' && pinkyState === 'CLOSE') {
    if (index_horiz && thumbState === 'OPEN') return 'G';  // horizontal + thumb out → G
    return 'D';                                            // pointing up → D
  }

  // F: index bent/closed, other three fingers open
  if ((indexState === 'CLOSE' || indexState === 'HALF_OPEN') && middleState === 'OPEN' && ringState === 'OPEN' && pinkyState === 'OPEN')
    return 'F';

  // X: only index hooked (HALF_OPEN), rest curled
  if (indexState === 'HALF_OPEN' && middleState === 'CLOSE' && ringState === 'CLOSE' && pinkyState === 'CLOSE')
    return 'X';

  // All four fingers HALF_OPEN — could be O (thumb near index) or C (curved, thumb away)
  if (indexState === 'HALF_OPEN' && middleState === 'HALF_OPEN' && ringState === 'HALF_OPEN' && pinkyState === 'HALF_OPEN') {
    if (d_t_i < 1.2 * ref) return 'O';  // thumb and index close together → O
    return 'C';                           // open curve → C
  }

  // N: index + middle hooked, ring + pinky curled
  if (indexState === 'HALF_OPEN' && middleState === 'HALF_OPEN' && ringState === 'CLOSE' && pinkyState === 'CLOSE')
    return 'N';

  // M: index, middle, ring hooked, pinky curled
  if (indexState === 'HALF_OPEN' && middleState === 'HALF_OPEN' && ringState === 'HALF_OPEN' && pinkyState === 'CLOSE')
    return 'M';

  // All four fingers fully curled (fist) — could be A, S, T, or E depending on thumb.
  if (indexState === 'CLOSE' && middleState === 'CLOSE' && ringState === 'CLOSE' && pinkyState === 'CLOSE') {
    if (thumbState === 'OPEN') return 'A';           // thumb sticking out → A
    if (d_t_i < 0.8 * ref) return 'S';              // thumb crosses over index → S
    if (d_t_m < 1.0 * ref) return 'T';              // thumb tucked under index touching middle → T
    return 'E';                                       // fingers bent, thumb tucked under → E
  }

  // No rule matched.
  return '';
}
