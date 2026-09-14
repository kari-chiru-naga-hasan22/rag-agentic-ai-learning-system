import React, { useState, useMemo } from 'react';
import { Compass, RefreshCw, Info, CheckCircle2 } from 'lucide-react';

export const VectorMathSandbox: React.FC = () => {
  const [ux, setUx] = useState<number>(3);
  const [uy, setUy] = useState<number>(4);
  const [vx, setVx] = useState<number>(4);
  const [vy, setVy] = useState<number>(2);

  const math = useMemo(() => {
    const dotProduct = ux * vx + uy * vy;
    const normU = Math.sqrt(ux * ux + uy * uy);
    const normV = Math.sqrt(vx * vx + vy * vy);
    const denominator = normU * normV;
    const cosSim = denominator === 0 ? 0 : Math.max(-1, Math.min(1, dotProduct / denominator));
    const angleRad = Math.acos(cosSim);
    const angleDeg = (angleRad * 180) / Math.PI;
    const euclideanDist = Math.sqrt(Math.pow(ux - vx, 2) + Math.pow(uy - vy, 2));

    // Unit normalized vectors
    const normUx = normU === 0 ? 0 : ux / normU;
    const normUy = normU === 0 ? 0 : uy / normU;
    const normVx = normV === 0 ? 0 : vx / normV;
    const normVy = normV === 0 ? 0 : vy / normV;
    const normalizedDot = normUx * normVx + normUy * normVy;

    return {
      dotProduct: dotProduct.toFixed(3),
      normU: normU.toFixed(3),
      normV: normV.toFixed(3),
      cosSim: cosSim.toFixed(4),
      angleDeg: angleDeg.toFixed(1),
      euclideanDist: euclideanDist.toFixed(3),
      normUx: normUx.toFixed(3),
      normUy: normUy.toFixed(3),
      normVx: normVx.toFixed(3),
      normVy: normVy.toFixed(3),
      normalizedDot: normalizedDot.toFixed(4),
    };
  }, [ux, uy, vx, vy]);

  const setPreset = (preset: 'orthogonal' | 'identical' | 'opposed' | 'close') => {
    if (preset === 'orthogonal') {
      setUx(0); setUy(4);
      setVx(4); setVy(0);
    } else if (preset === 'identical') {
      setUx(3); setUy(3);
      setVx(4.5); setVy(4.5);
    } else if (preset === 'opposed') {
      setUx(3); setUy(2);
      setVx(-3); setVy(-2);
    } else if (preset === 'close') {
      setUx(3); setUy(4);
      setVx(4); setVy(3);
    }
  };

  // SVG coordinate transform: canvas is 300x300, center is (150, 150), scale 25px per unit (span -5 to +5)
  const toSvgX = (x: number) => 150 + x * 24;
  const toSvgY = (y: number) => 150 - y * 24;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      <div className="border-b border-slate-100 pb-5 flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Compass className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">Vector Math & Cosine Similarity Sandbox</h2>
            <p className="text-sm text-slate-500">
              Interactive 2D geometric visualization of dot products, Euclidean norms, and angle metrics.
            </p>
          </div>
        </div>

        {/* Presets */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setPreset('orthogonal')}
            className="text-xs font-semibold px-2.5 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-indigo-50 hover:border-indigo-300 text-slate-700 hover:text-indigo-700 transition"
          >
            Orthogonal (90°)
          </button>
          <button
            onClick={() => setPreset('identical')}
            className="text-xs font-semibold px-2.5 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-indigo-50 hover:border-indigo-300 text-slate-700 hover:text-indigo-700 transition"
          >
            Collinear (0°)
          </button>
          <button
            onClick={() => setPreset('opposed')}
            className="text-xs font-semibold px-2.5 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-indigo-50 hover:border-indigo-300 text-slate-700 hover:text-indigo-700 transition"
          >
            Opposed (180°)
          </button>
          <button
            onClick={() => setPreset('close')}
            className="text-xs font-semibold px-2.5 py-1.5 rounded-lg border border-slate-200 bg-slate-50 hover:bg-indigo-50 hover:border-indigo-300 text-slate-700 hover:text-indigo-700 transition"
          >
            Close Semantic Match
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* SVG Plane & Sliders */}
        <div className="lg:col-span-6 space-y-6">
          <div className="flex justify-center p-4 bg-slate-50 rounded-2xl border border-slate-200">
            <svg width="300" height="300" className="overflow-visible select-none">
              {/* Grid lines */}
              {[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5].map((val) => (
                <g key={val}>
                  <line
                    x1={toSvgX(val)}
                    y1={10}
                    x2={toSvgX(val)}
                    y2={290}
                    stroke="#e2e8f0"
                    strokeWidth="1"
                  />
                  <line
                    x1={10}
                    y1={toSvgY(val)}
                    x2={290}
                    y2={toSvgY(val)}
                    stroke="#e2e8f0"
                    strokeWidth="1"
                  />
                </g>
              ))}

              {/* Main axes */}
              <line x1="150" y1="5" x2="150" y2="295" stroke="#94a3b8" strokeWidth="2" />
              <line x1="5" y1="150" x2="295" y2="150" stroke="#94a3b8" strokeWidth="2" />

              {/* Origin dot */}
              <circle cx="150" cy="150" r="3" fill="#64748b" />

              {/* Vector U (Indigo) */}
              <line
                x1="150"
                y1="150"
                x2={toSvgX(ux)}
                y2={toSvgY(uy)}
                stroke="#4f46e5"
                strokeWidth="3"
                strokeLinecap="round"
              />
              <circle cx={toSvgX(ux)} cy={toSvgY(uy)} r="5" fill="#4f46e5" />
              <text
                x={toSvgX(ux) + 8}
                y={toSvgY(uy) - 8}
                fill="#4f46e5"
                fontSize="12"
                fontWeight="bold"
              >
                u ({ux}, {uy})
              </text>

              {/* Vector V (Emerald) */}
              <line
                x1="150"
                y1="150"
                x2={toSvgX(vx)}
                y2={toSvgY(vy)}
                stroke="#059669"
                strokeWidth="3"
                strokeLinecap="round"
              />
              <circle cx={toSvgX(vx)} cy={toSvgY(vy)} r="5" fill="#059669" />
              <text
                x={toSvgX(vx) + 8}
                y={toSvgY(vy) + 14}
                fill="#059669"
                fontSize="12"
                fontWeight="bold"
              >
                v ({vx}, {vy})
              </text>
            </svg>
          </div>

          {/* Coordinate Sliders */}
          <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200">
            <div className="space-y-3">
              <span className="text-xs font-bold text-indigo-700 uppercase tracking-wider block">
                Vector u (Query Vector)
              </span>
              <div>
                <label className="text-xs text-slate-500 flex justify-between">
                  <span>u_x:</span> <strong>{ux}</strong>
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.5"
                  value={ux}
                  onChange={(e) => setUx(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded accent-indigo-600 cursor-pointer"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 flex justify-between">
                  <span>u_y:</span> <strong>{uy}</strong>
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.5"
                  value={uy}
                  onChange={(e) => setUy(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded accent-indigo-600 cursor-pointer"
                />
              </div>
            </div>

            <div className="space-y-3">
              <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider block">
                Vector v (Document Chunk)
              </span>
              <div>
                <label className="text-xs text-slate-500 flex justify-between">
                  <span>v_x:</span> <strong>{vx}</strong>
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.5"
                  value={vx}
                  onChange={(e) => setVx(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded accent-emerald-600 cursor-pointer"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 flex justify-between">
                  <span>v_y:</span> <strong>{vy}</strong>
                </label>
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.5"
                  value={vy}
                  onChange={(e) => setVy(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded accent-emerald-600 cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Live Mathematical Proof & Output */}
        <div className="lg:col-span-6 space-y-5">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Real-Time Metric Computation
          </h3>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-indigo-50/70 border border-indigo-100 p-4 rounded-xl">
              <div className="text-xs font-semibold text-indigo-700 uppercase">Cosine Similarity</div>
              <div className="text-2xl font-extrabold text-indigo-950 mt-1">{math.cosSim}</div>
              <div className="text-xs text-indigo-800/80 mt-1">Range: [-1.0, 1.0]</div>
            </div>

            <div className="bg-emerald-50/70 border border-emerald-100 p-4 rounded-xl">
              <div className="text-xs font-semibold text-emerald-700 uppercase">Angular Separation (θ)</div>
              <div className="text-2xl font-extrabold text-emerald-950 mt-1">{math.angleDeg}°</div>
              <div className="text-xs text-emerald-800/80 mt-1">cos(θ) = {math.cosSim}</div>
            </div>

            <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl">
              <div className="text-xs font-semibold text-slate-500 uppercase">Dot Product (u · v)</div>
              <div className="text-xl font-bold text-slate-800 mt-1">{math.dotProduct}</div>
              <div className="text-xs text-slate-500 mt-1">({ux}×{vx}) + ({uy}×{vy})</div>
            </div>

            <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl">
              <div className="text-xs font-semibold text-slate-500 uppercase">Euclidean Distance</div>
              <div className="text-xl font-bold text-slate-800 mt-1">{math.euclideanDist}</div>
              <div className="text-xs text-slate-500 mt-1">||u - v||₂ norm</div>
            </div>
          </div>

          {/* Derivation breakdown */}
          <div className="bg-slate-900 text-slate-100 p-5 rounded-xl font-mono text-xs space-y-2.5 overflow-x-auto">
            <div className="text-indigo-300 font-semibold">// Vector Norm Calculations</div>
            <div>||u||₂ = √({ux}² + {uy}²) = {math.normU}</div>
            <div>||v||₂ = √({vx}² + {vy}²) = {math.normV}</div>
            <div className="text-slate-500">------------------------------------------------</div>
            <div className="text-indigo-300 font-semibold">// Cosine Formula & Equivalence Proof</div>
            <div>S_C(u, v) = (u · v) / (||u||₂ × ||v||₂)</div>
            <div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {math.dotProduct} / ({math.normU} × {math.normV}) = {math.cosSim}</div>
            <div className="text-slate-500">------------------------------------------------</div>
            <div className="text-emerald-300 font-semibold">// Unit Normalization Equivalence: û · v̂</div>
            <div>û = [{math.normUx}, {math.normUy}]</div>
            <div>v̂ = [{math.normVx}, {math.normVy}]</div>
            <div>û · v̂ = ({math.normUx} × {math.normVx}) + ({math.normUy} × {math.normVy}) = {math.normalizedDot}</div>
          </div>

          <div className="p-3.5 bg-amber-50 border border-amber-200 rounded-xl flex items-start space-x-2.5 text-xs text-amber-900">
            <Info className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
            <div>
              <strong>Production Vector DB Insight:</strong> High-performance vector databases (Qdrant, Milvus, pgvector) normalize all embeddings to unit vectors (||v||₂ = 1) at ingestion time. This allows replacing expensive trigonometric cosine division with a pure BLAS dot product (u · v), saving 60% of SIMD instructions during HNSW graph traversal!
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
