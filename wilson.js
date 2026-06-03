/**
 * Ecuación de Wilson (1964) – Equilibrio Vapor–Líquido
 * ======================================================
 * Referencia principal:
 *   Wilson, G.M. (1964). A new expression for the excess free energy of mixing.
 *   Journal of the American Chemical Society, 86(2), 127–130.
 *   DOI: 10.1021/ja01056a002
 *
 * VLE con Ley de Raoult Modificada:
 *   yᵢ·P = xᵢ·γᵢ·Pᵢˢᵃᵗ(T)
 *
 * Presión de vapor (Ecuación de Antoine):
 *   log₁₀(Pˢᵃᵗ/mmHg) = A – B/(T/°C + C)
 *   Convertimos a kPa: Pˢᵃᵗ[kPa] = 10^(A–B/(T+C)) · 0.133322
 *
 * Fuentes de parámetros:
 *   – DECHEMA VLE Data Series, Gmehling et al. (1977–1988)
 *   – Smith, Van Ness & Abbott, "Introduction to Chemical Engineering
 *     Thermodynamics", 8th Ed. (2017), Appendix B
 *   – Perry's Chemical Engineers' Handbook, 8th Ed.
 */

'use strict';

/* ═══════════════════════════════════════════════════════
   DATABASE: Sistemas binarios con parámetros validados
   ═══════════════════════════════════════════════════════ */
const SYSTEMS = {
  'etanol-agua': {
    name1: 'Etanol',  name2: 'Agua',
    lambda12: 0.7241, lambda21: 0.4350,
    // Antoine (mmHg, °C): Perry's 8th Ed.
    ant1: { A: 8.1122, B: 1592.864, C: 226.184 },
    ant2: { A: 8.07131, B: 1730.630, C: 233.426 },
    note: 'DECHEMA VLE Data, Vol. 1, Pt. 1 (Gmehling et al., 1977)',
    hasAzeo: true
  },
  'metanol-agua': {
    name1: 'Metanol', name2: 'Agua',
    lambda12: 0.7252, lambda21: 0.5169,
    ant1: { A: 7.89750, B: 1474.08, C: 214.870 },
    ant2: { A: 8.07131, B: 1730.630, C: 233.426 },
    note: 'Smith, Van Ness & Abbott, 8th Ed., Ex. 13.1'
  },
  '1propanol-agua': {
    name1: '1-Propanol', name2: 'Agua',
    lambda12: 0.2150, lambda21: 0.7700,
    ant1: { A: 7.99733, B: 1569.68, C: 209.534 },
    ant2: { A: 8.07131, B: 1730.630, C: 233.426 },
    note: 'DECHEMA VLE Data, Vol. 1, Pt. 1 (Gmehling et al., 1977)',
    hasAzeo: true
  },
  'acetona-agua': {
    name1: 'Acetona', name2: 'Agua',
    lambda12: 0.2088, lambda21: 0.5622,
    ant1: { A: 7.11714, B: 1210.595, C: 229.664 },
    ant2: { A: 8.07131, B: 1730.630, C: 233.426 },
    note: 'DECHEMA VLE Data, Vol. 2, Pt. 1 (Gmehling & Onken, 1977)'
  },
  'etanol-acetona': {
    name1: 'Etanol', name2: 'Acetona',
    lambda12: 0.9237, lambda21: 1.0471,
    ant1: { A: 8.1122,  B: 1592.864, C: 226.184 },
    ant2: { A: 7.11714, B: 1210.595, C: 229.664 },
    note: 'Prausnitz, Lichtenthaler & de Azevedo (1999), App. A'
  },
  'metanol-acetona': {
    name1: 'Metanol', name2: 'Acetona',
    lambda12: 0.5755, lambda21: 0.6598,
    ant1: { A: 7.89750, B: 1474.08, C: 214.870 },
    ant2: { A: 7.11714, B: 1210.595, C: 229.664 },
    note: 'DECHEMA VLE Data, Vol. 2, Pt. 2 (Gmehling & Onken, 1977)'
  },
  'custom': {
    name1: 'Componente 1', name2: 'Componente 2',
    lambda12: 0.5, lambda21: 0.5,
    ant1: { A: 8.1122, B: 1592.864, C: 226.184 },
    ant2: { A: 8.07131, B: 1730.630, C: 233.426 },
    note: 'Parámetros definidos por el usuario'
  }
};

/* ═══════════════════════════════════════════════════════
   ESTADO DE LA APLICACIÓN
   ═══════════════════════════════════════════════════════ */
let state = {
  systemKey: 'etanol-agua',
  lambda12: 0.7241,
  lambda21: 0.4350,
  ant1: { A: 8.1122, B: 1592.864, C: 226.184 },
  ant2: { A: 8.07131, B: 1730.630, C: 233.426 },
  mode: 'pxy',   // 'pxy' o 'txy'
  T: 70,         // °C
  P: 101.325,    // kPa
  probeX: 0.5
};

/* ═══════════════════════════════════════════════════════
   TERMODINÁMICA – NÚCLEO MATEMÁTICO
   ═══════════════════════════════════════════════════════ */

/** Presión de vapor mediante Ecuación de Antoine
 *  log₁₀(P/mmHg) = A - B/(T°C + C)  →  kPa
 */
function pSat(T_C, ant) {
  const logP = ant.A - ant.B / (T_C + ant.C);
  return Math.pow(10, logP) * 0.133322; // mmHg → kPa
}

/** Ecuación de Wilson (1964) – coeficientes de actividad para mezcla binaria
 *  ln γ₁ = -ln(x₁ + Λ₁₂x₂) + x₂ · [Λ₁₂/(x₁+Λ₁₂x₂) – Λ₂₁/(Λ₂₁x₁+x₂)]
 *  ln γ₂ = -ln(x₂ + Λ₂₁x₁) + x₁ · [Λ₂₁/(Λ₂₁x₁+x₂) – Λ₁₂/(x₁+Λ₁₂x₂)]
 *
 *  Esta es la forma binaria explícita derivada del caso general multi-componente.
 *  Verificada contra Smith, Van Ness & Abbott (2017), Ec. 13.44–13.45.
 */
function wilsonGamma(x1, L12, L21) {
  const x2 = 1 - x1;
  const eps = 1e-12;
  const S1 = x1 + L12 * x2 + eps;
  const S2 = x2 + L21 * x1 + eps;

  const lnG1 = -Math.log(S1) + x2 * (L12 / S1 - L21 / S2);
  const lnG2 = -Math.log(S2) + x1 * (L21 / S2 - L12 / S1);

  return { g1: Math.exp(lnG1), g2: Math.exp(lnG2) };
}

/** Energía de Gibbs en Exceso normalizada por RT
 *  G^E/RT = -x₁·ln(x₁ + Λ₁₂x₂) - x₂·ln(x₂ + Λ₂₁x₁)
 *  Wilson (1964), Ec. 4
 */
function gibbs_excess_RT(x1, L12, L21) {
  const x2 = 1 - x1;
  const eps = 1e-12;
  return -x1 * Math.log(x1 + L12 * x2 + eps) - x2 * Math.log(x2 + L21 * x1 + eps);
}

/** Volatilidad relativa α₁₂ = (y₁/x₁)/(y₂/x₂) = γ₁·P₁ˢᵃᵗ/(γ₂·P₂ˢᵃᵗ) */
function relativeVolatility(x1, L12, L21, ps1, ps2) {
  const { g1, g2 } = wilsonGamma(x1, L12, L21);
  return (g1 * ps1) / (g2 * ps2 + 1e-15);
}

/** Fracción vapor a P fija (diagrama P-xy):
 *  P_burbuja = Σ xᵢ·γᵢ·Pᵢˢᵃᵗ
 *  y₁ = x₁·γ₁·P₁ˢᵃᵗ / P_burbuja
 */
function calcPxy(x1, L12, L21, T_C) {
  const { g1, g2 } = wilsonGamma(x1, L12, L21);
  const ps1 = pSat(T_C, state.ant1);
  const ps2 = pSat(T_C, state.ant2);
  const Pbub = x1 * g1 * ps1 + (1 - x1) * g2 * ps2;
  const y1   = (x1 * g1 * ps1) / (Pbub + 1e-15);
  return { P: Pbub, y1: Math.max(0, Math.min(1, y1)) };
}

/** Temperatura de burbuja a P fija (diagrama T-xy) usando bisección.
 *  Condición: Σ xᵢ·γᵢ·Pᵢˢᵃᵗ(T) = P
 *  γᵢ dependen de la composición (x), no de T directamente en la forma simple.
 */
function calcTbubble(x1, L12, L21, P_kPa) {
  const { g1, g2 } = wilsonGamma(x1, L12, L21);
  // Bisección en temperatura
  let lo = 20, hi = 200;
  for (let k = 0; k < 80; k++) {
    const mid = (lo + hi) / 2;
    const ps1 = pSat(mid, state.ant1);
    const ps2 = pSat(mid, state.ant2);
    const Pcalc = x1 * g1 * ps1 + (1 - x1) * g2 * ps2;
    if (Pcalc < P_kPa) lo = mid; else hi = mid;
  }
  const T = (lo + hi) / 2;
  const ps1 = pSat(T, state.ant1);
  const ps2 = pSat(T, state.ant2);
  const Pbub = x1 * g1 * ps1 + (1 - x1) * g2 * ps2;
  const y1 = (x1 * g1 * ps1) / (Pbub + 1e-15);
  return { T, y1: Math.max(0, Math.min(1, y1)) };
}

/** Temperatura de rocío a P fija */
function calcTdew(y1, L12, L21, P_kPa) {
  // x₁ inicial: Raoult ideal
  let x1_est = 0.5;
  let T = 80;
  // Iteración punto fijo
  for (let k = 0; k < 120; k++) {
    const { g1, g2 } = wilsonGamma(x1_est, L12, L21);
    // Bisección T dado x
    let lo = 20, hi = 200;
    for (let j = 0; j < 50; j++) {
      const mid = (lo + hi) / 2;
      const ps1 = pSat(mid, state.ant1);
      const ps2 = pSat(mid, state.ant2);
      const inv = y1 / (g1 * ps1) + (1 - y1) / (g2 * ps2);
      if (1 / inv < P_kPa) lo = mid; else hi = mid;
    }
    T = (lo + hi) / 2;
    const ps1 = pSat(T, state.ant1);
    const ps2 = pSat(T, state.ant2);
    const x1_new = y1 * P_kPa / (g1 * ps1 + 1e-15);
    x1_new < 0 || x1_new > 1 ? x1_est = 0.5 : x1_est = x1_new;
    if (Math.abs(x1_new - x1_est) < 1e-6) break;
  }
  return { T, x1: Math.max(0, Math.min(1, x1_est)) };
}

/** Detección de azeótropo: busca x₁ donde y₁ = x₁ (dP/dx₁ = 0 o dT/dx₁ = 0) */
function findAzeotrope(L12, L21) {
  // Buscamos cambio de signo en (y₁ - x₁) para modo P-xy a T fija
  const T = state.T;
  let prev_diff = null, azeo = null;
  for (let i = 1; i < 199; i++) {
    const x1 = i / 200;
    const { y1 } = calcPxy(x1, L12, L21, T);
    const diff = y1 - x1;
    if (prev_diff !== null && prev_diff * diff < 0) {
      // Refinamiento por bisección
      let lo = (i - 1) / 200, hi = i / 200;
      for (let k = 0; k < 40; k++) {
        const mid = (lo + hi) / 2;
        const d_mid = calcPxy(mid, L12, L21, T).y1 - mid;
        if (prev_diff * d_mid < 0) hi = mid; else { lo = mid; prev_diff = d_mid; }
      }
      const xAz = (lo + hi) / 2;
      const { P: PAz } = calcPxy(xAz, L12, L21, T);
      azeo = { x1: xAz, P: PAz };
      break;
    }
    prev_diff = diff;
  }
  return azeo;
}

/* ═══════════════════════════════════════════════════════
   CANVAS RENDERERS
   ═══════════════════════════════════════════════════════ */

const C1_COLOR  = '#58d4ff';
const C2_COLOR  = '#ff8fc8';
const GE_COLOR  = '#80ffb0';
const IDEAL_CLR = 'rgba(200,200,255,0.25)';
const N_POINTS  = 300;

/** Utilidad: mapeo de valor a píxel */
function mkScale(vmin, vmax, pmin, pmax) {
  return v => pmin + (v - vmin) / (vmax - vmin) * (pmax - pmin);
}

/** Dibuja ejes, grid y devuelve funciones de escala */
function drawAxes(ctx, W, H, pad, xRange, yRange, xlbl, ylbl) {
  // Background
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#020818');
  bg.addColorStop(1, '#040a22');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const sx = mkScale(xRange[0], xRange[1], pad.l, W - pad.r);
  const sy = mkScale(yRange[0], yRange[1], H - pad.b, pad.t);

  // Grid
  ctx.strokeStyle = 'rgba(60,100,200,0.13)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const xv = xRange[0] + i * (xRange[1] - xRange[0]) / 5;
    const yv = yRange[0] + i * (yRange[1] - yRange[0]) / 5;
    const px = sx(xv), py = sy(yv);
    ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, H - pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(W - pad.r, py); ctx.stroke();

    ctx.fillStyle = '#8898cc';
    ctx.font = '10px Nunito, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xv.toFixed(1), px, H - pad.b + 13);
    ctx.textAlign = 'right';
    ctx.fillText(yv.toFixed(yRange[1] < 5 ? 2 : 0), pad.l - 4, py + 4);
  }

  // Axis labels
  ctx.fillStyle = '#8898cc'; ctx.font = '11px Nunito,sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xlbl, (pad.l + W - pad.r) / 2, H - 3);
  ctx.save();
  ctx.translate(11, (pad.t + H - pad.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylbl, 0, 0);
  ctx.restore();

  return { sx, sy };
}

/* ── VLE Diagram (P-xy or T-xy) ── */
function drawVLE() {
  const canvas = document.getElementById('vleCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { l: 55, r: 20, t: 22, b: 40 };

  // Collect data
  const xs = [], ys_bub = [], ys_dew = [], ys_ideal = [];
  let yMin = Infinity, yMax = -Infinity;

  const L12 = state.lambda12, L21 = state.lambda21;

  for (let i = 0; i <= N_POINTS; i++) {
    const x1 = i / N_POINTS;
    xs.push(x1);

    let bub, dew_y, ideal_y;
    if (state.mode === 'pxy') {
      const res = calcPxy(x1, L12, L21, state.T);
      bub = res.P;
      dew_y = res.y1;
      // Ideal (Raoult puro)
      const ps1 = pSat(state.T, state.ant1);
      const ps2 = pSat(state.T, state.ant2);
      ideal_y = x1 * ps1 + (1 - x1) * ps2;
    } else {
      const res = calcTbubble(x1, L12, L21, state.P);
      bub = res.T;
      dew_y = res.y1;
      const ps1i = pSat(res.T, state.ant1);
      const ps2i = pSat(res.T, state.ant2);
      ideal_y = res.T; // para la línea de Raoult ideal calculamos T a Raoult puro
    }

    ys_bub.push(bub);
    ys_ideal.push(ideal_y);
    yMin = Math.min(yMin, bub);
    yMax = Math.max(yMax, bub);
  }

  // Dew point curve: invert (y1 → P o T)
  const dew_pts = [];
  for (let i = 0; i <= N_POINTS; i++) {
    const x1 = i / N_POINTS;
    if (state.mode === 'pxy') {
      // For dew: find x1_dew such that y1_bubble(x1_dew) = x1 (this x1 is now y1)
      // Simple: sweep x and get y
      const res = calcPxy(x1, L12, L21, state.T);
      dew_pts.push({ y1: res.y1, P: res.P });
    } else {
      const res = calcTbubble(x1, L12, L21, state.P);
      dew_pts.push({ y1: res.y1, T: res.T });
    }
  }

  const margin = (yMax - yMin) * 0.06 || 5;
  const yRange = [Math.max(0, yMin - margin), yMax + margin];
  const xRange = [0, 1];

  const yLabel = state.mode === 'pxy' ? 'P (kPa)' : 'T (°C)';
  const { sx, sy } = drawAxes(ctx, W, H, pad, xRange, yRange,
    state.mode === 'pxy' ? 'Fracción molar x₁, y₁' : 'Fracción molar x₁, y₁', yLabel);

  // Ideal line
  if (state.mode === 'pxy') {
    ctx.beginPath();
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = IDEAL_CLR;
    ctx.lineWidth = 1.5;
    ys_ideal.forEach((v, i) => {
      const px = sx(xs[i]), py = sy(v);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Dew curve (y₁ vs P or T): use bubble data sorted by y₁
  // Build dew envelope: for each unique y1, find P from bubble
  const dew_sorted = [...dew_pts].sort((a, b) => a.y1 - b.y1);
  ctx.beginPath();
  ctx.strokeStyle = C2_COLOR;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = C2_COLOR; ctx.shadowBlur = 8;
  let first = true;
  dew_sorted.forEach(pt => {
    const val = state.mode === 'pxy' ? pt.P : pt.T;
    if (isNaN(val) || val < yRange[0] || val > yRange[1]) return;
    const px = sx(pt.y1), py = sy(val);
    if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Bubble curve (x₁ vs P or T)
  ctx.beginPath();
  ctx.strokeStyle = C1_COLOR;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = C1_COLOR; ctx.shadowBlur = 8;
  ys_bub.forEach((v, i) => {
    if (isNaN(v)) return;
    const px = sx(xs[i]), py = sy(v);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Azeotrope marker
  const azeo = findAzeotrope(L12, L21);
  if (azeo && azeo.x1 > 0.01 && azeo.x1 < 0.99) {
    const azV = state.mode === 'pxy' ? azeo.P : calcTbubble(azeo.x1, L12, L21, state.P).T;
    const ax = sx(azeo.x1), ay = sy(azV);
    // Diamond marker
    ctx.save();
    ctx.translate(ax, ay);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = '#ffd700';
    ctx.shadowColor = '#ffd700'; ctx.shadowBlur = 14;
    ctx.fillRect(-6, -6, 12, 12);
    ctx.shadowBlur = 0;
    ctx.restore();
    ctx.fillStyle = '#ffd700'; ctx.font = 'bold 10px Nunito,sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Azeo: x₁=${azeo.x1.toFixed(3)}`, ax, ay - 14);

    // Update azeotrope display
    document.getElementById('azeo-icon').textContent = '⚠️';
    document.getElementById('azeo-title').textContent = 'Azeótropo detectado';
    document.getElementById('azeo-val').textContent =
      `x₁ = y₁ = ${azeo.x1.toFixed(4)} · P = ${azeo.P.toFixed(3)} kPa`;
    document.getElementById('azeotrope-box').style.borderColor = 'rgba(255,215,0,0.4)';
  } else {
    document.getElementById('azeo-icon').textContent = '✅';
    document.getElementById('azeo-title').textContent = 'Sin azeótropo';
    document.getElementById('azeo-val').textContent = 'Sistema zeótropo (no forma azeótropo)';
    document.getElementById('azeotrope-box').style.borderColor = '';
  }

  // Probe vertical line
  const probeX = sx(state.probeX);
  ctx.strokeStyle = 'rgba(255,255,200,0.35)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(probeX, pad.t); ctx.lineTo(probeX, H - pad.b); ctx.stroke();
  ctx.setLineDash([]);

  // Labels in plot area
  ctx.fillStyle = C1_COLOR; ctx.font = 'bold 10px Nunito,sans-serif'; ctx.textAlign = 'left';
  ctx.fillText('← Burbuja (x₁)', pad.l + 6, pad.t + 16);
  ctx.fillStyle = C2_COLOR;
  ctx.fillText('← Rocío (y₁)', pad.l + 6, pad.t + 30);
}

/* ── Activity Coefficients ── */
function drawGamma() {
  const canvas = document.getElementById('gammaCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { l: 48, r: 16, t: 18, b: 36 };

  const g1s = [], g2s = [], xs = [];
  const L12 = state.lambda12, L21 = state.lambda21;
  let gMax = 1;

  for (let i = 0; i <= N_POINTS; i++) {
    const x1 = i / N_POINTS;
    const { g1, g2 } = wilsonGamma(x1, L12, L21);
    xs.push(x1);
    g1s.push(g1);
    g2s.push(g2);
    gMax = Math.max(gMax, g1, g2);
  }

  const { sx, sy } = drawAxes(ctx, W, H, pad, [0, 1], [0.9, gMax + 0.1],
    'x₁', 'γᵢ');

  // γ₁ = 1 reference
  ctx.strokeStyle = 'rgba(200,200,255,0.2)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  const y1line = sy(1);
  ctx.beginPath(); ctx.moveTo(pad.l, y1line); ctx.lineTo(W - pad.r, y1line); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(200,200,255,0.4)'; ctx.font = '9px Nunito,sans-serif';
  ctx.textAlign = 'left'; ctx.fillText('γ=1 (Raoult)', pad.l + 3, y1line - 4);

  // γ₂
  ctx.beginPath();
  ctx.strokeStyle = C2_COLOR; ctx.lineWidth = 2.2;
  ctx.shadowColor = C2_COLOR; ctx.shadowBlur = 6;
  g2s.forEach((v, i) => {
    const px = sx(xs[i]), py = sy(v);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
  // γ₁
  ctx.beginPath();
  ctx.strokeStyle = C1_COLOR; ctx.lineWidth = 2.2;
  ctx.shadowColor = C1_COLOR; ctx.shadowBlur = 6;
  g1s.forEach((v, i) => {
    const px = sx(xs[i]), py = sy(v);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Probe dot
  const { g1: pg1, g2: pg2 } = wilsonGamma(state.probeX, L12, L21);
  [[pg1, C1_COLOR], [pg2, C2_COLOR]].forEach(([gv, col]) => {
    const px = sx(state.probeX), py = sy(gv);
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 10;
    ctx.fill(); ctx.shadowBlur = 0;
  });

  // Labels
  ctx.fillStyle = C1_COLOR; ctx.font = 'bold 9px Nunito,sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('γ₁', sx(0.1), sy(g1s[Math.round(0.1 * N_POINTS)]) - 6);
  ctx.fillStyle = C2_COLOR;
  ctx.fillText('γ₂', sx(0.9), sy(g2s[Math.round(0.9 * N_POINTS)]) - 6);
}

/* ── Gibbs Excess Energy ── */
function drawGE() {
  const canvas = document.getElementById('geCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { l: 52, r: 16, t: 18, b: 36 };

  const ges = [], xs = [];
  const L12 = state.lambda12, L21 = state.lambda21;
  let geMax = 0, geMin = 0;

  for (let i = 1; i < N_POINTS; i++) {
    const x1 = i / N_POINTS;
    const ge = gibbs_excess_RT(x1, L12, L21);
    xs.push(x1);
    ges.push(ge);
    geMax = Math.max(geMax, ge);
    geMin = Math.min(geMin, ge);
  }

  const margin = (geMax - geMin) * 0.1 || 0.05;
  const { sx, sy } = drawAxes(ctx, W, H, pad, [0, 1],
    [geMin - margin, geMax + margin], 'x₁', 'G^E/RT');

  // Zero line
  ctx.strokeStyle = 'rgba(200,200,255,0.18)'; ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, sy(0)); ctx.lineTo(W - pad.r, sy(0));
  ctx.stroke(); ctx.setLineDash([]);

  // GE/RT curve
  ctx.beginPath();
  ctx.strokeStyle = GE_COLOR; ctx.lineWidth = 2.4;
  ctx.shadowColor = GE_COLOR; ctx.shadowBlur = 8;
  ges.forEach((v, i) => {
    const px = sx(xs[i]), py = sy(v);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Fill under curve
  ctx.beginPath();
  ctx.moveTo(sx(xs[0]), sy(0));
  ges.forEach((v, i) => ctx.lineTo(sx(xs[i]), sy(v)));
  ctx.lineTo(sx(xs[xs.length-1]), sy(0));
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, sy(geMax), 0, sy(0));
  grad.addColorStop(0, 'rgba(128,255,176,0.22)');
  grad.addColorStop(1, 'rgba(128,255,176,0.02)');
  ctx.fillStyle = grad;
  ctx.fill();

  // Maximum label
  const maxIdx = ges.indexOf(geMax);
  if (maxIdx >= 0) {
    const px = sx(xs[maxIdx]), py = sy(geMax);
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = GE_COLOR; ctx.shadowColor = GE_COLOR; ctx.shadowBlur = 10;
    ctx.fill(); ctx.shadowBlur = 0;
    ctx.fillStyle = GE_COLOR; ctx.font = 'bold 9px Nunito,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`max: ${geMax.toFixed(4)}`, px, py - 8);
  }

  // Update metrics
  document.getElementById('met-ge-max').textContent = geMax.toFixed(5);
}

/* ═══════════════════════════════════════════════════════
   PROBE CALCULATIONS
   ═══════════════════════════════════════════════════════ */
function updateProbe() {
  const x1 = state.probeX;
  const L12 = state.lambda12, L21 = state.lambda21;
  const { g1, g2 } = wilsonGamma(x1, L12, L21);
  const ge = gibbs_excess_RT(x1, L12, L21);
  const ps1 = pSat(state.T, state.ant1);
  const ps2 = pSat(state.T, state.ant2);
  const { P, y1 } = calcPxy(x1, L12, L21, state.T);
  const alpha = relativeVolatility(x1, L12, L21, ps1, ps2);

  // Panel
  document.getElementById('pr-g1').textContent = g1.toFixed(5);
  document.getElementById('pr-g2').textContent = g2.toFixed(5);
  document.getElementById('pr-y1').textContent = y1.toFixed(5);
  document.getElementById('pr-y2').textContent = (1 - y1).toFixed(5);
  document.getElementById('pr-ge').textContent = ge.toFixed(6);
  document.getElementById('pr-alpha').textContent = alpha.toFixed(4);

  // Top metrics
  document.getElementById('met-g1').textContent = g1.toFixed(5);
  document.getElementById('met-g2').textContent = g2.toFixed(5);
  document.getElementById('met-psat').textContent =
    `${ps1.toFixed(2)} / ${ps2.toFixed(2)} kPa`;
}

/* ═══════════════════════════════════════════════════════
   RENDER ALL
   ═══════════════════════════════════════════════════════ */
function renderAll() {
  drawVLE();
  drawGamma();
  drawGE();
  updateProbe();
}

/* ═══════════════════════════════════════════════════════
   SYSTEM SELECTOR
   ═══════════════════════════════════════════════════════ */
function applySystem(key) {
  const sys = SYSTEMS[key];
  if (!sys) return;
  state.systemKey = key;
  state.lambda12  = sys.lambda12;
  state.lambda21  = sys.lambda21;
  state.ant1 = { ...sys.ant1 };
  state.ant2 = { ...sys.ant2 };

  // Update UI
  document.getElementById('comp1-name').textContent = sys.name1;
  document.getElementById('comp2-name').textContent = sys.name2;
  document.getElementById('lambda12').value = sys.lambda12;
  document.getElementById('lambda21').value = sys.lambda21;
  document.getElementById('lambda12-val').textContent = sys.lambda12.toFixed(4);
  document.getElementById('lambda21-val').textContent = sys.lambda21.toFixed(4);
  document.getElementById('params-note').textContent = sys.note || '';

  ['A','B','C'].forEach(k => {
    document.getElementById(`ant-A1`).value = sys.ant1.A;
    document.getElementById(`ant-B1`).value = sys.ant1.B;
    document.getElementById(`ant-C1`).value = sys.ant1.C;
    document.getElementById(`ant-A2`).value = sys.ant2.A;
    document.getElementById(`ant-B2`).value = sys.ant2.B;
    document.getElementById(`ant-C2`).value = sys.ant2.C;
  });

  renderAll();
}

/* ═══════════════════════════════════════════════════════
   EVENT LISTENERS
   ═══════════════════════════════════════════════════════ */

// System selector
document.getElementById('system-select').addEventListener('change', e => {
  applySystem(e.target.value);
});

// Wilson parameters
['lambda12','lambda21'].forEach(id => {
  document.getElementById(id).addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    state[id] = v;
    document.getElementById(`${id}-val`).textContent = v.toFixed(4);
    // Switch to custom
    document.getElementById('system-select').value = 'custom';
    renderAll();
  });
});

// Temperature slider (P-xy mode)
document.getElementById('temp-val-sl').addEventListener('input', e => {
  state.T = parseFloat(e.target.value);
  document.getElementById('temp-display').textContent = `${state.T} °C`;
  renderAll();
});

// Pressure slider (T-xy mode)
document.getElementById('press-val-sl').addEventListener('input', e => {
  state.P = parseFloat(e.target.value);
  document.getElementById('press-display').textContent = `${state.P.toFixed(1)} kPa`;
  renderAll();
});

// Mode toggle
document.getElementById('mode-pxy').addEventListener('click', () => {
  state.mode = 'pxy';
  document.getElementById('mode-pxy').classList.add('active');
  document.getElementById('mode-txy').classList.remove('active');
  document.getElementById('cond-T').style.display = '';
  document.getElementById('cond-P').style.display = 'none';
  document.getElementById('vle-title').textContent = '📊 Diagrama P-xy (T fija)';
  renderAll();
});
document.getElementById('mode-txy').addEventListener('click', () => {
  state.mode = 'txy';
  document.getElementById('mode-txy').classList.add('active');
  document.getElementById('mode-pxy').classList.remove('active');
  document.getElementById('cond-T').style.display = 'none';
  document.getElementById('cond-P').style.display = '';
  document.getElementById('vle-title').textContent = '🌡️ Diagrama T-xy (P fija)';
  renderAll();
});

// Probe slider
document.getElementById('probe-x').addEventListener('input', e => {
  state.probeX = parseFloat(e.target.value);
  document.getElementById('probe-x-val').textContent = state.probeX.toFixed(3);
  renderAll();
});

// Antoine inputs
['A1','B1','C1','A2','B2','C2'].forEach(k => {
  const el = document.getElementById(`ant-${k}`);
  if (!el) return;
  el.addEventListener('input', e => {
    const comp = k.endsWith('1') ? 'ant1' : 'ant2';
    const par  = k.slice(0, 1); // A, B, or C
    state[comp][par] = parseFloat(e.target.value) || 0;
    document.getElementById('system-select').value = 'custom';
    renderAll();
  });
});

// Canvas click for probe x in VLE
document.getElementById('vleCanvas').addEventListener('click', e => {
  const rect = document.getElementById('vleCanvas').getBoundingClientRect();
  const scaleX = document.getElementById('vleCanvas').width / rect.width;
  const cx = (e.clientX - rect.left) * scaleX;
  const pad = { l: 55, r: 20 };
  const W = document.getElementById('vleCanvas').width;
  const innerW = W - pad.l - pad.r;
  const x1 = Math.max(0.001, Math.min(0.999, (cx - pad.l) / innerW));
  state.probeX = x1;
  document.getElementById('probe-x').value = x1;
  document.getElementById('probe-x-val').textContent = x1.toFixed(3);
  renderAll();
});

/* ═══════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════ */
function resizeCanvases() {
  ['vleCanvas','gammaCanvas','geCanvas'].forEach(id => {
    const canvas = document.getElementById(id);
    const wrap   = canvas.parentElement;
    const w = wrap.clientWidth;
    if (w < 10) return;
    const aspectMap = { vleCanvas: 580/380, gammaCanvas: 280/230, geCanvas: 280/230 };
    const aspect = aspectMap[id] || 1.5;
    canvas.width  = Math.round(w * 2);
    canvas.height = Math.round(w * 2 / aspect);
  });
  renderAll();
}
window.addEventListener('resize', () => {
  clearTimeout(window._rTimer);
  window._rTimer = setTimeout(resizeCanvases, 150);
});

/* ═══════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════ */
window.addEventListener('load', () => {
  setTimeout(() => {
    applySystem('etanol-agua');
    resizeCanvases();
  }, 120);
});
