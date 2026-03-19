import { useMemo, useState } from 'react';
import type { DataPoint } from '../types';
import { useTheme } from '../context/ThemeContext';
import { getTwClasses, getTheme } from '../theme';
import {
  float32ToDX7Params,
  getCarriers,
  ALGORITHMS,
  LFO_WAVEFORM_NAMES,
} from '../dx7';
import type { DX7Params, DX7OperatorParams, AlgorithmDef } from '../dx7';

type TwClasses = ReturnType<typeof getTwClasses>;

// ---------------------------------------------------------------------------
// AlgorithmDiagram
// ---------------------------------------------------------------------------

/**
 * Build a tree of operators rooted at each carrier for layout purposes.
 * Returns an array of trees — one per carrier — where each node knows its
 * children (the operators that modulate it, excluding self-feedback).
 */
interface OpNode {
  op: number;
  children: OpNode[];
}

function buildModulatorTrees(algDef: AlgorithmDef): OpNode[] {
  const assigned = new Set<number>();
  const carrierList = [...algDef.outputMix].sort((a, b) => a - b);

  function buildSubtree(op: number): OpNode {
    assigned.add(op);
    const children: OpNode[] = [];
    for (const mod of algDef.modulationMatrix[op]) {
      if (mod !== op && !assigned.has(mod)) {
        children.push(buildSubtree(mod));
      }
    }
    return { op, children };
  }

  const trees: OpNode[] = [];
  for (const c of carrierList) {
    if (!assigned.has(c)) {
      trees.push(buildSubtree(c));
    }
  }
  // Pick up any ops not reachable from carriers
  for (let i = 0; i < 6; i++) {
    if (!assigned.has(i)) {
      trees.push(buildSubtree(i));
    }
  }
  return trees;
}

/** Count the leaf-width of a tree (minimum columns it needs). */
function treeWidth(node: OpNode): number {
  if (node.children.length === 0) return 1;
  return node.children.reduce((sum, c) => sum + treeWidth(c), 0);
}

/** Recursively assign x,y positions within a horizontal span. */
function layoutTree(
  node: OpNode,
  xMin: number,
  xMax: number,
  y: number,
  rowHeight: number,
  positions: { x: number; y: number }[],
) {
  positions[node.op] = { x: (xMin + xMax) / 2, y };

  if (node.children.length === 0) return;

  const totalW = node.children.reduce((s, c) => s + treeWidth(c), 0);
  const span = xMax - xMin;
  let cx = xMin;

  for (const child of node.children) {
    const w = treeWidth(child);
    const childSpan = (w / totalW) * span;
    layoutTree(child, cx, cx + childSpan, y - rowHeight, rowHeight, positions);
    cx += childSpan;
  }
}

/** Layout positions for operators in the algorithm diagram SVG. */
function computeOpPositions(algDef: AlgorithmDef): { x: number; y: number }[] {
  const positions: { x: number; y: number }[] = Array.from({ length: 6 }, () => ({ x: 0, y: 0 }));

  const trees = buildModulatorTrees(algDef);
  const totalW = trees.reduce((s, t) => s + treeWidth(t), 0);

  const svgWidth = 180;
  const padding = 10;
  const usable = svgWidth - padding * 2;
  let x = padding;

  for (const tree of trees) {
    const w = treeWidth(tree);
    const span = (w / totalW) * usable;
    layoutTree(tree, x, x + span, 120, 30, positions);
    x += span;
  }

  return positions;
}

interface AlgorithmDiagramProps {
  algorithm: number; // 1-indexed
  feedback: number;
  themeColors: ReturnType<typeof getTheme>;
}

function AlgorithmDiagram({ algorithm, feedback, themeColors }: AlgorithmDiagramProps) {
  const algIdx = algorithm - 1;
  if (algIdx < 0 || algIdx >= ALGORITHMS.length) return null;
  const algDef = ALGORITHMS[algIdx];
  const carrierSet = new Set(algDef.outputMix);
  const positions = computeOpPositions(algDef);
  const r = 9;

  // Find the feedback operator: the op that modulates itself
  // In DX7, the last op in a chain typically has self-feedback,
  // but we derive it from the modulation matrix
  let feedbackOp = -1;
  if (feedback > 0) {
    for (let i = 0; i < 6; i++) {
      if (algDef.modulationMatrix[i].includes(i)) {
        feedbackOp = i;
        break;
      }
    }
    // If no explicit self-modulation in matrix, feedback is on last modulator in chain
    // For DX7, feedback is always on op6 (index 5) by convention
    if (feedbackOp === -1) {
      feedbackOp = 5;
    }
  }

  return (
    <svg viewBox="0 0 180 140" width="180" height="140" className="block mx-auto">
      {/* Draw modulation arrows */}
      {Array.from({ length: 6 }, (_, target) =>
        algDef.modulationMatrix[target].map((mod) => {
          if (mod === target) return null; // self-feedback drawn separately
          const from = positions[mod];
          const to = positions[target];
          const dx = to.x - from.x;
          const dy = to.y - from.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist === 0) return null;
          const nx = dx / dist;
          const ny = dy / dist;
          const x1 = from.x + nx * r;
          const y1 = from.y + ny * r;
          const x2 = to.x - nx * r;
          const y2 = to.y - ny * r;
          return (
            <line
              key={`${mod}-${target}`}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={themeColors.preset.lighter}
              strokeWidth={1.2}
              markerEnd="url(#arrowhead)"
            />
          );
        })
      )}

      {/* Feedback arc */}
      {feedbackOp >= 0 && feedback > 0 && (
        <path
          d={`M ${positions[feedbackOp].x + r} ${positions[feedbackOp].y - 4}
              C ${positions[feedbackOp].x + r + 14} ${positions[feedbackOp].y - 16}
                ${positions[feedbackOp].x + r + 14} ${positions[feedbackOp].y + 8}
                ${positions[feedbackOp].x + r} ${positions[feedbackOp].y + 4}`}
          fill="none"
          stroke={themeColors.warning.light}
          strokeWidth={1}
          markerEnd="url(#arrowhead-fb)"
        />
      )}

      {/* Operator circles */}
      {positions.map((pos, i) => {
        const isCarrier = carrierSet.has(i);
        return (
          <g key={i}>
            <circle
              cx={pos.x} cy={pos.y} r={r}
              fill={isCarrier ? themeColors.audio.bg : themeColors.preset.bg}
              stroke={isCarrier ? themeColors.audio.primary : themeColors.preset.primary}
              strokeWidth={1.5}
            />
            <text
              x={pos.x} y={pos.y + 3.5}
              textAnchor="middle"
              fontSize="8"
              fontWeight="600"
              fill={isCarrier ? themeColors.audio.light : themeColors.preset.light}
            >
              {i + 1}
            </text>
          </g>
        );
      })}

      {/* Arrow marker defs */}
      <defs>
        <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
          <polygon points="0 0, 6 2, 0 4" fill={themeColors.preset.lighter} />
        </marker>
        <marker id="arrowhead-fb" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
          <polygon points="0 0, 6 2, 0 4" fill={themeColors.warning.light} />
        </marker>
      </defs>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// MiniEnvelope
// ---------------------------------------------------------------------------

interface MiniEnvelopeProps {
  rates: [number, number, number, number];
  levels: [number, number, number, number];
  themeColors: ReturnType<typeof getTheme>;
}

function MiniEnvelope({ rates, levels, themeColors }: MiniEnvelopeProps) {
  // Convert DX7 rates/levels to x,y polyline points in a 60x24 viewBox
  // rates control time (higher = faster), levels control amplitude
  // Envelope stages: L0->L1 at R0, L1->L2 at R1, L2->L3 at R2, sustain at L3, L3->0 at R3
  const maxTime = 60;
  const maxLevel = 24;

  // Convert rate to time duration (higher rate = shorter time)
  const rateToTime = (r: number) => Math.max(1, ((99 - r) / 99) * 13);

  const points: string[] = [];
  let x = 0;
  // Start at level 0
  const l0y = maxLevel - (levels[0] / 99) * maxLevel;
  points.push(`0,${maxLevel}`); // start from bottom
  points.push(`0,${l0y}`); // instant jump to L0 (attack start)

  for (let i = 0; i < 4; i++) {
    const dt = rateToTime(rates[i]);
    x += dt;
    const ly = maxLevel - (levels[i] / 99) * maxLevel;
    points.push(`${Math.min(x, maxTime)},${ly}`);
  }

  const strokeColor = themeColors.audio.lighter;

  return (
    <svg viewBox="0 0 60 24" width="52" height="20" className="block">
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke={strokeColor}
        strokeWidth={1.2}
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// OperatorStrip
// ---------------------------------------------------------------------------

interface OperatorStripProps {
  op: DX7OperatorParams;
  opIndex: number;
  isCarrier: boolean;
  twClasses: TwClasses;
  themeColors: ReturnType<typeof getTheme>;
}

function OperatorStrip({ op, opIndex, isCarrier, twClasses, themeColors }: OperatorStripProps) {
  const barHeight = Math.max(1, (op.volume / 99) * 40);
  const barColor = isCarrier ? themeColors.audio.primary : themeColors.preset.primary;
  const bgTint = isCarrier ? 'bg-blue-500/5' : 'bg-orange-500/5';

  // Frequency display
  let freqStr: string;
  if (op.oscMode === 1) {
    freqStr = 'FIX';
  } else {
    const coarse = op.freqCoarse === 0 ? 0.5 : op.freqCoarse;
    const fine = 1 + op.freqFine / 100;
    const ratio = coarse * fine;
    freqStr = ratio.toFixed(2);
  }

  return (
    <div className={`flex flex-col items-center gap-0.5 px-1 py-1 rounded ${bgTint}`} style={{ minWidth: 48 }}>
      {/* Op label */}
      <span className={`text-[9px] font-bold ${isCarrier ? twClasses.audioText : twClasses.presetText}`}>
        OP{opIndex + 1}
      </span>

      {/* Output level bar */}
      <div className="relative w-2 bg-black/20 rounded-sm" style={{ height: 40 }}>
        <div
          className="absolute bottom-0 w-full rounded-sm transition-all"
          style={{ height: barHeight, backgroundColor: barColor }}
        />
      </div>
      <span className={`text-[8px] font-mono ${twClasses.textMuted}`}>
        {op.volume}
      </span>

      {/* Frequency */}
      <span className={`text-[8px] font-mono ${twClasses.textSecondary}`}>
        {freqStr}
      </span>

      {/* Mini envelope */}
      <MiniEnvelope
        rates={op.rates}
        levels={op.levels}
        themeColors={themeColors}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// GlobalSection
// ---------------------------------------------------------------------------

interface GlobalSectionProps {
  params: DX7Params;
  twClasses: TwClasses;
  themeColors: ReturnType<typeof getTheme>;
}

function GlobalSection({ params, twClasses, themeColors }: GlobalSectionProps) {
  return (
    <div className={`px-3 py-1.5 border-t ${twClasses.borderSubtle}`}>
      <div className="flex items-start gap-x-1 text-[10px] font-mono">
        {[
          { label: 'LFO spd', value: params.lfoSpeed },
          { label: 'wave', value: LFO_WAVEFORM_NAMES[params.lfoWaveform] ?? '?' },
          { label: 'PMS', value: params.lfoPitchModSens },
          { label: 'FB', value: params.feedback },
          { label: 'TRS', value: params.transpose },
        ].map(({ label, value }) => (
          <div key={label} className="flex flex-col items-center" style={{ minWidth: 56 }}>
            <span className={twClasses.textSubtle}>{label}</span>
            <span className={twClasses.textSecondary}>{value}</span>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 mt-1">
        <span className={`text-[10px] font-mono ${twClasses.textMuted}`}>
          Pitch EG
        </span>
        <MiniEnvelope
          rates={params.pitchEnvelope.rates}
          levels={params.pitchEnvelope.levels}
          themeColors={themeColors}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PresetHeader
// ---------------------------------------------------------------------------

interface PresetHeaderProps {
  point: DataPoint;
  params: DX7Params;
  minimized: boolean;
  onToggleMinimize: () => void;
  onClose: () => void;
  twClasses: TwClasses;
}

function PresetHeader({ point, params, minimized, onToggleMinimize, onClose, twClasses }: PresetHeaderProps) {
  const badge = point.modality === 'audio' ? twClasses.audioBadge : twClasses.presetBadge;
  const modalityLabel = point.modality === 'audio' ? 'audio' : 'preset';

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 border-b ${twClasses.borderSubtle}`}>
      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${badge}`}>
        {modalityLabel}
      </span>
      <span className={`text-sm font-semibold ${twClasses.textPrimary} truncate`}>
        {params.name}
      </span>
      <span className={`text-[10px] ${twClasses.textSubtle} ml-auto mr-2 flex-shrink-0`}>
        {point.pair_id}
      </span>
      <button
        onClick={onToggleMinimize}
        className={`${twClasses.buttonGhost} text-sm leading-none px-1 flex-shrink-0`}
        title={minimized ? 'Expand' : 'Minimize'}
      >
        {minimized ? '+' : '\u2212'}
      </button>
      <button
        onClick={onClose}
        className={`${twClasses.buttonGhost} text-lg leading-none flex-shrink-0`}
        title="Close"
      >
        {'\u00d7'}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DX7PresetPanel (main export)
// ---------------------------------------------------------------------------

interface DX7PresetPanelProps {
  selectedPoint: DataPoint;
  hoveredPoint: DataPoint | null;
  onClose: () => void;
}

export function DX7PresetPanel({ selectedPoint, hoveredPoint, onClose }: DX7PresetPanelProps) {
  const { mode } = useTheme();
  const twClasses = getTwClasses(mode);
  const themeColors = getTheme(mode);
  const [minimized, setMinimized] = useState(false);

  // Show hovered point's preset when hovering, otherwise show selected
  const displayPoint = (hoveredPoint && hoveredPoint.id !== selectedPoint.id)
    ? hoveredPoint
    : selectedPoint;

  const params = useMemo(
    () => float32ToDX7Params(displayPoint.dx7_preset, displayPoint.preset_name),
    [displayPoint.dx7_preset, displayPoint.preset_name]
  );

  const algIdx = params.algorithm - 1;
  const carrierSet = new Set(algIdx >= 0 && algIdx < ALGORITHMS.length ? getCarriers(algIdx) : []);

  return (
    <div
      className={`absolute ${twClasses.bgElevatedSemi} backdrop-blur-sm rounded-lg shadow-lg border ${twClasses.borderSubtle}`}
      style={{ right: 16, top: 4, width: 380, zIndex: 10 }}
    >
      {/* Header */}
      <PresetHeader
        point={displayPoint}
        params={params}
        minimized={minimized}
        onToggleMinimize={() => setMinimized((m) => !m)}
        onClose={onClose}
        twClasses={twClasses}
      />

      {minimized ? null : (
        <>
          {/* Algorithm section */}
          <div className={`px-3 py-2 border-b ${twClasses.borderSubtle}`}>
            <div className="flex items-center justify-between mb-1">
              <span className={`text-xs font-semibold ${twClasses.textSecondary}`}>
                Algorithm {params.algorithm}
              </span>
              <span className={`text-xs ${twClasses.textMuted}`}>
                Feedback: {params.feedback}
              </span>
            </div>
            <AlgorithmDiagram
              algorithm={params.algorithm}
              feedback={params.feedback}
              themeColors={themeColors}
            />
          </div>

          {/* Operators section */}
          <div className={`px-2 py-2 border-b ${twClasses.borderSubtle}`}>
            <div className="flex justify-between">
              {params.operators.map((op, i) => (
                <OperatorStrip
                  key={i}
                  op={op}
                  opIndex={i}
                  isCarrier={carrierSet.has(i)}
                  twClasses={twClasses}
                  themeColors={themeColors}
                />
              ))}
            </div>
          </div>

          {/* Global / LFO section */}
          <GlobalSection
            params={params}
            twClasses={twClasses}
            themeColors={themeColors}
          />
        </>
      )}
    </div>
  );
}
