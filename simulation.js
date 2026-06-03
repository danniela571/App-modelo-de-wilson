/**
 * Wilson-Cowan Neural Population Model – Interactive Simulation
 * ================================================================
 * Based on: Wilson, H.R. & Cowan, J.D. (1972). "Excitatory and inhibitory
 * interactions in localized populations of model neurons."
 * Biophysical Journal, 12(1), 1–24. DOI:10.1016/S0006-3495(72)86068-5
 *
 * Model Equations:
 *   τ_E · dE/dt = -E + f_E(w_EE·E - w_EI·I + P_E)
 *   τ_I · dI/dt = -I + f_I(w_IE·E - w_II·I + P_I)
 *   f(x) = 1 / (1 + exp(-a·(x - θ)))   [sigmoid activation]
 *
 * Integration: 4th-order Runge-Kutta (RK4)
 */

'use strict';

/* ── Canvas references ── */
const phaseCanvas   = document.getElementById('phaseCanvas');
const timeCanvas    = document.getElementById('timeCanvas');
const sigmoidCanvas = document.getElementById('sigmoidCanvas');
const phaseCtx      = phaseCanvas.getContext('2d');
const timeCtx       = timeCanvas.getContext('2d');
const sigCtx        = sigmoidCanvas.getContext('2d');

/* ── Model state ── */
let state = {
  E: 0.2,
  I: 0.2,
  t: 0
};

/* ── Parameters (with scientific defaults – oscillation regime) ── */
let params = {
  wEE: 12,   wEI: 4,
  wIE: 13,   wII: 11,
  PE:  -2,   PI:  -3.5,
  tauE: 1,   tauI: 1,
  sigA: 1.2, sigTheta: 2.8,
  E0: 0.2,   I0: 0.2,
  dt: 0.05
};

/* ── Simulation runtime ── */
let isRunning  = false;
let animFrameId = null;
let history    = { E: [], I: [], t: [] };
const MAX_HISTORY = 800;

/* ═══════════════════════════════════════════════════════════════
   PRESET CONFIGURATIONS (scientifically validated parameter sets)
   Based on Borisyuk & Kirillov (1992) and Deco et al. (2008)
   ═══════════════════════════════════════════════════════════════ */
const PRESETS = {
  oscillation: {
    wEE: 12, wEI: 4,  wIE: 13, wII: 11,
    PE: -2,  PI: -3.5,
    tauE: 1, tauI: 1,
    sigA: 1.2, sigTheta: 2.8,
    E0: 0.2, I0: 0.2,
    label: '🌀 Oscilación'
  },
  bistable: {
    wEE: 10, wEI: 2,  wIE: 6,  wII: 1,
    PE: -0.5, PI: -2,
    tauE: 1,  tauI: 1,
    sigA: 1.5, sigTheta: 2.0,
    E0: 0.05, I0: 0.05,
    label: '⚡ Bistable'
  },
  excitable: {
    wEE: 5,  wEI: 3,  wIE: 8,  wII: 2,
    PE: -1,  PI: -2,
    tauE: 1, tauI: 2,
    sigA: 1.0, sigTheta: 3.0,
    E0: 0.05, I0: 0.05,
    label: '🔔 Excitable'
  },
  rest: {
    wEE: 3,  wEI: 5,  wIE: 4,  wII: 2,
    PE: -3,  PI: -4,
    tauE: 1, tauI: 1,
    sigA: 1.0, sigTheta: 4.0,
    E0: 0.01, I0: 0.01,
    label: '💤 Reposo'
  }
};

/* ═══════════════════════════════════════════════════════════════
   MATH CORE
   ═══════════════════════════════════════════════════════════════ */

/** Sigmoid activation function */
function sigmoid(x, a, theta) {
  return 1 / (1 + Math.exp(-a * (x - theta)));
}

/** Wilson-Cowan ODEs – returns [dE, dI] */
function wilsonCowanODE(E, I, p) {
  const { wEE, wEI, wIE, wII, PE, PI, tauE, tauI, sigA, sigTheta } = p;
  const fe = sigmoid(wEE * E - wEI * I + PE, sigA, sigTheta);
  const fi = sigmoid(wIE * E - wII * I + PI, sigA, sigTheta);
  const dE = (-E + fe) / tauE;
  const dI = (-I + fi) / tauI;
  return [dE, dI];
}

/** 4th-order Runge-Kutta integration step */
function rk4Step(E, I, dt, p) {
  const [k1E, k1I] = wilsonCowanODE(E, I, p);
  const [k2E, k2I] = wilsonCowanODE(E + 0.5 * dt * k1E, I + 0.5 * dt * k1I, p);
  const [k3E, k3I] = wilsonCowanODE(E + 0.5 * dt * k2E, I + 0.5 * dt * k2I, p);
  const [k4E, k4I] = wilsonCowanODE(E + dt * k3E, I + dt * k3I, p);
  const newE = E + (dt / 6) * (k1E + 2 * k2E + 2 * k3E + k4E);
  const newI = I + (dt / 6) * (k1I + 2 * k2I + 2 * k3I + k4I);
  return [newE, newI];
}

/** Clamp value to [0, 1] (physiological constraint) */
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

/* ═══════════════════════════════════════════════════════════════
   NULLCLINE COMPUTATION
   Nullclines are the curves where dE/dt = 0 and dI/dt = 0
   Used for phase plane visualization (qualitative analysis)
   ═══════════════════════════════════════════════════════════════ */

/** E-nullcline: E = f_E(w_EE·E - w_EI·I + P_E)
    Solve for I given E: I = (w_EE·E + P_E - sigInv(E)) / w_EI */
function eNullclineI(E, p) {
  const { wEE, wEI, PE, sigA, sigTheta } = p;
  if (E <= 0.001 || E >= 0.999) return NaN;
  const invSig = -Math.log(1 / E - 1) / sigA + sigTheta;
  if (wEI === 0) return NaN;
  return (wEE * E + PE - invSig) / wEI;
}

/** I-nullcline: I = f_I(w_IE·E - w_II·I + P_I)
    Solve for I given E using Newton's method */
function iNullclineI(E, p) {
  const { wIE, wII, PI, sigA, sigTheta } = p;
  if (wII === 0) {
    return sigmoid(wIE * E + PI, sigA, sigTheta);
  }
  // Solve I = sigmoid(wIE*E - wII*I + PI) iteratively
  let I = 0.5;
  for (let k = 0; k < 60; k++) {
    const rhs = sigmoid(wIE * E - wII * I + PI, sigA, sigTheta);
    const deriv = -wII * sigA * rhs * (1 - rhs);
    const f = I - rhs;
    const df = 1 - deriv;
    const dI = f / df;
    I -= dI;
    I = clamp(I);
    if (Math.abs(dI) < 1e-9) break;
  }
  return I;
}

/* ═══════════════════════════════════════════════════════════════
   FIXED POINT DETECTION
   Finds intersections of nullclines (equilibrium points)
   ═══════════════════════════════════════════════════════════════ */
function findFixedPoints(p) {
  const fps = [];
  const Evals = [];
  for (let i = 0; i <= 200; i++) Evals.push(i / 200);

  for (let i = 0; i < Evals.length - 1; i++) {
    const E0 = Evals[i];
    const E1 = Evals[i + 1];
    const iE0 = eNullclineI(E0, p);
    const iE1 = eNullclineI(E1, p);
    const iI0 = iNullclineI(E0, p);
    const iI1 = iNullclineI(E1, p);
    if (isNaN(iE0) || isNaN(iE1)) continue;

    const d0 = iE0 - iI0;
    const d1 = iE1 - iI1;
    if (d0 * d1 <= 0) {
      // Sign change → candidate fixed point; refine by bisection
      let lo = E0, hi = E1;
      for (let k = 0; k < 40; k++) {
        const mid = (lo + hi) / 2;
        const dm = eNullclineI(mid, p) - iNullclineI(mid, p);
        if (isNaN(dm)) break;
        if (d0 * dm <= 0) hi = mid; else lo = mid;
      }
      const Efp = (lo + hi) / 2;
      const Ifp = iNullclineI(Efp, p);
      // De-duplicate
      if (!fps.some(fp => Math.abs(fp[0] - Efp) < 0.01)) {
        fps.push([Efp, clamp(Ifp)]);
      }
    }
  }
  return fps;
}

/** Classify stability of a fixed point using Jacobian eigenvalues */
function classifyFixedPoint(E, I, p) {
  const { wEE, wEI, wIE, wII, tauE, tauI, sigA, sigTheta } = p;
  const feIn = wEE * E - wEI * I + p.PE;
  const fiIn = wIE * E - wII * I + p.PI;
  const dfE = sigA * sigmoid(feIn, sigA, sigTheta) * (1 - sigmoid(feIn, sigA, sigTheta));
  const dfI = sigA * sigmoid(fiIn, sigA, sigTheta) * (1 - sigmoid(fiIn, sigA, sigTheta));
  // Jacobian: J = [(-1 + wEE*dfE)/tauE, -wEI*dfE/tauE;
  //                wIE*dfI/tauI,          (-1 - wII*dfI)/tauI]
  const j11 = (-1 + wEE * dfE) / tauE;
  const j12 = (-wEI * dfE) / tauE;
  const j21 = (wIE * dfI) / tauI;
  const j22 = (-1 - wII * dfI) / tauI;
  const tr  = j11 + j22;
  const det = j11 * j22 - j12 * j21;
  const disc = tr * tr - 4 * det;
  if (det < 0)  return 'saddle';
  if (tr > 0)   return 'unstable';
  if (disc < 0) return 'spiral';
  return 'stable';
}

/* ═══════════════════════════════════════════════════════════════
   REGIME DETECTOR (heuristic, for display)
   ═══════════════════════════════════════════════════════════════ */
function detectRegime(fps, histE) {
  if (fps.length === 0) return '—';
  if (fps.length >= 3)  return '⚡ Bistable';
  if (fps.length === 1) {
    const stab = classifyFixedPoint(fps[0][0], fps[0][1], params);
    if (stab === 'unstable' || stab === 'spiral') {
      // Check oscillation amplitude
      if (histE.length > 40) {
        const recent = histE.slice(-40);
        const amp = Math.max(...recent) - Math.min(...recent);
        if (amp > 0.05) return '🌀 Ciclo Límite';
      }
      return '🌀 Ciclo Límite';
    }
    if (stab === 'stable') return '💤 Punto Fijo Estable';
    return '🔔 Excitable';
  }
  return '🌀 Oscilatorio';
}

/* ═══════════════════════════════════════════════════════════════
   PHASE PLANE RENDERER
   ═══════════════════════════════════════════════════════════════ */
const PHASE_PAD = { l: 45, r: 18, t: 18, b: 38 };

function phaseToCanvas(E, I) {
  const W = phaseCanvas.width - PHASE_PAD.l - PHASE_PAD.r;
  const H = phaseCanvas.height - PHASE_PAD.t - PHASE_PAD.b;
  return [
    PHASE_PAD.l + E * W,
    PHASE_PAD.t + (1 - I) * H
  ];
}

function canvasToPhase(cx, cy) {
  const W = phaseCanvas.width - PHASE_PAD.l - PHASE_PAD.r;
  const H = phaseCanvas.height - PHASE_PAD.t - PHASE_PAD.b;
  return [
    clamp((cx - PHASE_PAD.l) / W),
    clamp(1 - (cy - PHASE_PAD.t) / H)
  ];
}

function drawPhase() {
  const ctx = phaseCtx;
  const W = phaseCanvas.width;
  const H = phaseCanvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#020818');
  bg.addColorStop(1, '#040a22');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(60, 100, 200, 0.15)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const [x0] = phaseToCanvas(i / 10, 0);
    const [, y0] = phaseToCanvas(0, i / 10);
    ctx.beginPath(); ctx.moveTo(x0, PHASE_PAD.t); ctx.lineTo(x0, H - PHASE_PAD.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PHASE_PAD.l, y0); ctx.lineTo(W - PHASE_PAD.r, y0); ctx.stroke();
  }

  // Axes labels
  ctx.fillStyle = '#8898cc';
  ctx.font = '11px Nunito, sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    const [x] = phaseToCanvas(v, 0);
    const [, y] = phaseToCanvas(0, v);
    ctx.fillText(v.toFixed(1), x, H - PHASE_PAD.b + 14);
    ctx.fillText(v.toFixed(1), PHASE_PAD.l - 8, y + 4);
  }
  ctx.fillText('E (Excitatorio)', W / 2, H - 4);
  ctx.save();
  ctx.translate(12, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('I (Inhibitorio)', 0, 0);
  ctx.restore();

  const N = 240; // nullcline resolution

  // ── E-Nullcline ──
  ctx.beginPath();
  ctx.strokeStyle = '#58b4ff';
  ctx.lineWidth = 2.2;
  ctx.shadowColor = '#58b4ff';
  ctx.shadowBlur = 8;
  let started = false;
  for (let k = 0; k <= N; k++) {
    const E = k / N;
    const I = eNullclineI(E, params);
    if (isNaN(I) || I < 0 || I > 1) { started = false; continue; }
    const [cx, cy] = phaseToCanvas(E, I);
    if (!started) { ctx.moveTo(cx, cy); started = true; }
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // ── I-Nullcline ──
  ctx.beginPath();
  ctx.strokeStyle = '#c57bff';
  ctx.lineWidth = 2.2;
  ctx.shadowColor = '#c57bff';
  ctx.shadowBlur = 8;
  started = false;
  for (let k = 0; k <= N; k++) {
    const E = k / N;
    const I = iNullclineI(E, params);
    if (isNaN(I) || I < 0 || I > 1) { started = false; continue; }
    const [cx, cy] = phaseToCanvas(E, I);
    if (!started) { ctx.moveTo(cx, cy); started = true; }
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // ── Vector field (arrows) ──
  const VN = 14;
  for (let i = 0; i <= VN; i++) {
    for (let j = 0; j <= VN; j++) {
      const E = i / VN;
      const I = j / VN;
      const [dE, dI] = wilsonCowanODE(E, I, params);
      const mag = Math.sqrt(dE * dE + dI * dI);
      if (mag < 1e-6) continue;
      const scale = 0.028;
      const dEn = (dE / mag) * scale;
      const dIn = (dI / mag) * scale;
      const [x0, y0] = phaseToCanvas(E, I);
      const [x1, y1] = phaseToCanvas(E + dEn, I + dIn);
      const alpha = Math.min(0.4, mag * 0.4);
      ctx.strokeStyle = `rgba(120, 160, 255, ${alpha})`;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      // Arrowhead
      const angle = Math.atan2(y1 - y0, x1 - x0);
      const ahLen = 4;
      ctx.fillStyle = `rgba(120, 160, 255, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - ahLen * Math.cos(angle - 0.4), y1 - ahLen * Math.sin(angle - 0.4));
      ctx.lineTo(x1 - ahLen * Math.cos(angle + 0.4), y1 - ahLen * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fill();
    }
  }

  // ── Trajectory ──
  if (history.E.length > 1) {
    const n = history.E.length;
    for (let k = 1; k < n; k++) {
      const alpha = 0.3 + 0.7 * (k / n);
      ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.7})`;
      ctx.lineWidth = 1.4;
      const [x0, y0] = phaseToCanvas(history.E[k - 1], history.I[k - 1]);
      const [x1, y1] = phaseToCanvas(history.E[k], history.I[k]);
      ctx.beginPath();
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    // Current position – animated dot
    const [cx, cy] = phaseToCanvas(state.E, state.I);
    const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 200);
    ctx.beginPath();
    ctx.arc(cx, cy, 5 + pulse * 2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255, 255, 255, ${0.7 + pulse * 0.3})`;
    ctx.shadowColor = '#fff';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // ── Fixed Points ──
  const fps = findFixedPoints(params);
  fps.forEach(([Efp, Ifp]) => {
    const stab = classifyFixedPoint(Efp, Ifp, params);
    const [cx, cy] = phaseToCanvas(Efp, Ifp);
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = stab === 'stable' ? '#ffd700' : 'rgba(255,100,100,0.9)';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = stab === 'stable' ? '#ffd700' : '#ff6464';
    ctx.shadowBlur = 14;
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 9px Nunito, sans-serif';
    ctx.textAlign = 'left';
    const labels = { stable: 'Estable', unstable: 'Inestable', saddle: 'Silla', spiral: 'Espiral' };
    ctx.fillText(labels[stab] || stab, cx + 10, cy - 4);
  });

  // Update regime display
  const regime = detectRegime(fps, history.E);
  document.getElementById('met-regime').textContent = regime;
}

/* ═══════════════════════════════════════════════════════════════
   TIME SERIES RENDERER
   ═══════════════════════════════════════════════════════════════ */
const TIME_PAD = { l: 45, r: 18, t: 18, b: 32 };

function drawTimeSeries() {
  const ctx = timeCtx;
  const W = timeCanvas.width;
  const H = timeCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#020818');
  bg.addColorStop(1, '#040a22');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(60, 100, 200, 0.12)';
  ctx.lineWidth = 1;
  for (let j = 0; j <= 4; j++) {
    const y = TIME_PAD.t + (H - TIME_PAD.t - TIME_PAD.b) * (j / 4);
    ctx.beginPath(); ctx.moveTo(TIME_PAD.l, y); ctx.lineTo(W - TIME_PAD.r, y); ctx.stroke();
    ctx.fillStyle = '#8898cc';
    ctx.font = '10px Nunito, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText((1 - j / 4).toFixed(1), TIME_PAD.l - 4, y + 4);
  }
  ctx.fillStyle = '#8898cc';
  ctx.font = '10px Nunito, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Actividad Neuronal', W / 2, H - 4);

  if (history.E.length < 2) return;

  const n = history.E.length;
  const iw = W - TIME_PAD.l - TIME_PAD.r;
  const ih = H - TIME_PAD.t - TIME_PAD.b;

  function valToY(v) {
    return TIME_PAD.t + (1 - v) * ih;
  }
  function idxToX(i) {
    return TIME_PAD.l + (i / (MAX_HISTORY - 1)) * iw;
  }

  // Draw E
  ctx.beginPath();
  ctx.strokeStyle = '#58b4ff';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#58b4ff';
  ctx.shadowBlur = 6;
  history.E.forEach((v, i) => {
    const x = idxToX(i);
    const y = valToY(clamp(v));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Draw I
  ctx.beginPath();
  ctx.strokeStyle = '#c57bff';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#c57bff';
  ctx.shadowBlur = 6;
  history.I.forEach((v, i) => {
    const x = idxToX(i);
    const y = valToY(clamp(v));
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Time axis label (approx. total time shown)
  const totalT = (history.t[history.t.length - 1] || 0).toFixed(0);
  ctx.fillStyle = '#8898cc';
  ctx.font = '10px Nunito, sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(`t=${totalT} ms`, W - TIME_PAD.r, H - TIME_PAD.b + 12);
}

/* ═══════════════════════════════════════════════════════════════
   SIGMOID VIEWER
   ═══════════════════════════════════════════════════════════════ */
function drawSigmoid() {
  const ctx = sigCtx;
  const W = sigmoidCanvas.width;
  const H = sigmoidCanvas.height;
  const pad = { l: 40, r: 16, t: 16, b: 30 };

  ctx.clearRect(0, 0, W, H);
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#020818');
  bg.addColorStop(1, '#040a22');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const xRange = [-6, 10];
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  function mapX(x) { return pad.l + (x - xRange[0]) / (xRange[1] - xRange[0]) * iw; }
  function mapY(y) { return pad.t + (1 - y) * ih; }

  // Grid
  ctx.strokeStyle = 'rgba(60, 100, 200, 0.15)';
  ctx.lineWidth = 1;
  for (let v = 0; v <= 1; v += 0.25) {
    const y = mapY(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#8898cc'; ctx.font = '10px Nunito, sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), pad.l - 4, y + 4);
  }
  // Zero line
  const xZero = mapX(0);
  ctx.strokeStyle = 'rgba(100, 140, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(xZero, pad.t); ctx.lineTo(xZero, H - pad.b); ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = '#8898cc'; ctx.font = '10px Nunito, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('0', xZero, H - pad.b + 14);
  ctx.fillText('Entrada Sináptica x', W / 2, H - 2);

  // Sigmoid curve
  ctx.beginPath();
  ctx.strokeStyle = '#7fc8ff';
  ctx.lineWidth = 2.5;
  ctx.shadowColor = '#7fc8ff';
  ctx.shadowBlur = 8;
  for (let px = 0; px <= iw; px++) {
    const x = xRange[0] + (px / iw) * (xRange[1] - xRange[0]);
    const y = sigmoid(x, params.sigA, params.sigTheta);
    const cx = pad.l + px;
    const cy = mapY(y);
    px === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Threshold line
  const xTh = mapX(params.sigTheta);
  if (xTh > pad.l && xTh < W - pad.r) {
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(xTh, pad.t); ctx.lineTo(xTh, H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffd700'; ctx.font = '10px Nunito, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`θ=${params.sigTheta.toFixed(1)}`, xTh, pad.t + 12);
  }

  // Midpoint dot (f(θ) = 0.5)
  const ymid = mapY(0.5);
  ctx.beginPath();
  ctx.arc(xTh, ymid, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#ffd700';
  ctx.shadowColor = '#ffd700';
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.shadowBlur = 0;
}

/* ═══════════════════════════════════════════════════════════════
   SIMULATION LOOP
   ═══════════════════════════════════════════════════════════════ */
const STEPS_PER_FRAME = 5; // Integration steps per animation frame

function simStep() {
  for (let k = 0; k < STEPS_PER_FRAME; k++) {
    const [newE, newI] = rk4Step(state.E, state.I, params.dt, params);
    state.E = clamp(newE);
    state.I = clamp(newI);
    state.t += params.dt;
    history.E.push(state.E);
    history.I.push(state.I);
    history.t.push(state.t);
    if (history.E.length > MAX_HISTORY) {
      history.E.shift();
      history.I.shift();
      history.t.shift();
    }
  }
}

function updateMetrics() {
  document.getElementById('met-E').textContent    = state.E.toFixed(4);
  document.getElementById('met-I').textContent    = state.I.toFixed(4);
  document.getElementById('met-time').textContent = state.t.toFixed(1);
}

function renderAll() {
  drawPhase();
  drawTimeSeries();
  drawSigmoid();
  updateMetrics();
}

function loop() {
  if (!isRunning) return;
  simStep();
  renderAll();
  animFrameId = requestAnimationFrame(loop);
}

/* ═══════════════════════════════════════════════════════════════
   CONTROLS
   ═══════════════════════════════════════════════════════════════ */

function startSim() {
  isRunning = true;
  document.getElementById('btn-run').textContent = '⏸ Pausar';
  document.getElementById('btn-run').classList.add('running');
  document.getElementById('status-dot').className = 'status-dot running';
  document.getElementById('status-text').textContent = 'Simulando…';
  loop();
}

function pauseSim() {
  isRunning = false;
  cancelAnimationFrame(animFrameId);
  document.getElementById('btn-run').textContent = '▶ Iniciar';
  document.getElementById('btn-run').classList.remove('running');
  document.getElementById('status-dot').className = 'status-dot paused';
  document.getElementById('status-text').textContent = 'Pausado';
}

function resetSim() {
  pauseSim();
  state.E = params.E0;
  state.I = params.I0;
  state.t = 0;
  history = { E: [], I: [], t: [] };
  document.getElementById('status-dot').className = 'status-dot';
  document.getElementById('status-text').textContent = 'Listo para simular';
  renderAll();
}

document.getElementById('btn-run').addEventListener('click', () => {
  isRunning ? pauseSim() : startSim();
});

document.getElementById('btn-reset').addEventListener('click', resetSim);

/* Phase plane click – set initial conditions */
phaseCanvas.addEventListener('click', (e) => {
  const rect = phaseCanvas.getBoundingClientRect();
  const scaleX = phaseCanvas.width  / rect.width;
  const scaleY = phaseCanvas.height / rect.height;
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top)  * scaleY;
  const [E, I] = canvasToPhase(cx, cy);
  // Reset history but keep time
  history = { E: [], I: [], t: [] };
  state.E = E;
  state.I = I;
  params.E0 = E;
  params.I0 = I;
  // Update sliders
  document.getElementById('E0').value = E.toFixed(2);
  document.getElementById('I0').value = I.toFixed(2);
  updateSliderVal('E0', E);
  updateSliderVal('I0', I);
  renderAll();
});

/* ═══════════════════════════════════════════════════════════════
   SLIDERS
   ═══════════════════════════════════════════════════════════════ */
const SLIDER_MAP = {
  wEE: 'wEE', wEI: 'wEI', wIE: 'wIE', wII: 'wII',
  PE: 'PE', PI: 'PI',
  tauE: 'tauE', tauI: 'tauI',
  sigA: 'sigA', sigTheta: 'sigTheta',
  E0: 'E0', I0: 'I0'
};

function updateSliderVal(id, val) {
  const display = document.getElementById(`${id}-val`);
  if (!display) return;
  const num = parseFloat(val);
  display.textContent = Number.isInteger(num * 10) ? num.toFixed(1) : num.toFixed(2);
}

Object.keys(SLIDER_MAP).forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;

  // Initialize display
  updateSliderVal(id, el.value);

  el.addEventListener('input', () => {
    const val = parseFloat(el.value);
    params[SLIDER_MAP[id]] = val;
    updateSliderVal(id, val);
    if (!isRunning) renderAll();
  });
});

/* ═══════════════════════════════════════════════════════════════
   PRESETS
   ═══════════════════════════════════════════════════════════════ */
function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;

  // Update params
  Object.assign(params, p);

  // Update sliders
  Object.keys(SLIDER_MAP).forEach(id => {
    const el = document.getElementById(id);
    const key = SLIDER_MAP[id];
    if (el && params[key] !== undefined) {
      el.value = params[key];
      updateSliderVal(id, params[key]);
    }
  });

  // Highlight active preset button
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`preset-${name}`).classList.add('active');

  // Reset simulation
  const wasRunning = isRunning;
  resetSim();
  if (wasRunning) startSim();
  else renderAll();
}

document.getElementById('preset-oscillation').addEventListener('click', () => applyPreset('oscillation'));
document.getElementById('preset-bistable').addEventListener('click',    () => applyPreset('bistable'));
document.getElementById('preset-excitable').addEventListener('click',   () => applyPreset('excitable'));
document.getElementById('preset-rest').addEventListener('click',        () => applyPreset('rest'));

/* ═══════════════════════════════════════════════════════════════
   RESPONSIVE CANVAS RESIZE
   ═══════════════════════════════════════════════════════════════ */
function resizeCanvases() {
  const phaseWrap = phaseCanvas.parentElement;
  const timeWrap  = timeCanvas.parentElement;
  const w = phaseWrap.clientWidth;
  if (w < 10) return;

  const phAspect = 560 / 420;
  const tmAspect = 560 / 220;
  phaseCanvas.width  = Math.round(w * 2);  // high-DPI
  phaseCanvas.height = Math.round(w * 2 / phAspect);
  timeCanvas.width   = Math.round(timeWrap.clientWidth * 2);
  timeCanvas.height  = Math.round(timeWrap.clientWidth * 2 / tmAspect);

  if (!isRunning) renderAll();
}

window.addEventListener('resize', () => {
  clearTimeout(window._resizeTimer);
  window._resizeTimer = setTimeout(resizeCanvases, 150);
});

/* ═══════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════ */
function init() {
  state.E = params.E0;
  state.I = params.I0;
  renderAll();

  // Auto-start after a brief delay for visual effect
  setTimeout(() => {
    startSim();
  }, 600);
}

// Wait for fonts/layout
window.addEventListener('load', () => {
  setTimeout(init, 100);
});
