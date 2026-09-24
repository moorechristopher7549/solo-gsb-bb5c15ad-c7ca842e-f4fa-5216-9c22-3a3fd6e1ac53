import type { AudioBufferLike } from '../types';

/**
 * 重复录制定位（纯函数，主线程与 Web Worker 共用；不触碰 AudioBuffer / DOM）。
 *
 * 以选中章节的原始音频为「样本」，在整段录音里寻找重录位置。匹配对整体
 * 音量变化不敏感（全部打分都用去均值归一化相关，Pearson r ∈ [-1, 1]）。
 *
 * 三级流水线（逐级加细，保证长音频也可在 Worker 中算完）：
 * 1. 能量粗筛：对降采样混合声道算 50ms 分块 RMS 能量轮廓，以样本能量
 *    轮廓与全体候选位置做归一化相关，粗筛能量走势一致的位置；
 * 2. 混合波形粗筛：在能量候选上对 200Hz 降采样混合波形做归一化相关，
 *    步长 5ms，把候选收敛到对齐位置附近；
 * 3. 归一化相关精匹配：在每个粗筛命中的 ±10ms 邻域，对 4kHz 混合波形、
 *    按样本头/中/尾取至多 3 段探针做归一化相关，相关值平均得到最终相似度。
 *
 * 样本自身区间及与样本重叠的候选一律排除；同一片重复段附近的多个命中
 * 按样本长度的一半为间隔合并，只保留相似度最高的代表。
 */

// ---- 调参常量（命名导出，便于测试与复用）----
/** 精匹配与粗波形使用的混合波形采样率（降采样）。 */
export const FINE_RATE = 4000;
/** 粗筛混合波形采样率（步长 5ms）。 */
export const COARSE_WAVE_RATE = 200;
/** 能量轮廓分块长度（秒）。 */
export const ENERGY_BLOCK_SEC = 0.05;
/** 样本最短可搜长度（秒）：太短的样本判别力不足。 */
export const MIN_SAMPLE_SEC = 0.5;
/** 探针最大长度（秒）；超过该时长的样本按头 / 中 / 尾取三段探针。 */
export const PROBE_MAX_SEC = 8;
/** 长样本最多取几段探针。 */
export const MAX_PROBES = 3;
/** 能量粗筛阈值（Pearson r）。 */
export const ENERGY_CUTOFF = 0.2;
/** 能量粗筛后进入波形粗筛的候选上限（取分最高者）。 */
export const ENERGY_MAX_CANDIDATES = 200;
/** 波形粗筛阈值（Pearson r）。 */
export const COARSE_CUTOFF = 0.45;
/** 波形粗筛后进入精匹配的候选上限。 */
export const COARSE_MAX_CANDIDATES = 24;
/** 精匹配相似度阈值（探针平均 Pearson r）。 */
export const FINE_CUTOFF = 0.6;
/** 探针 RMS 低于该值视为静音，不计入平均（全静音样本判空）。 */
export const PROBE_SILENCE_RMS = 1e-4;
/** 去均值平方和低于该值视为常值段（相关无定义）。 */
export const ZERO_VARIANCE_EPS = 1e-12;
/** 最终输出命中数量上限。 */
export const MAX_HITS = 50;

export interface RepeatSearchSource {
  duration: number;
  length: number;
  sampleRate: number;
  /** 全部声道（仅前两个声道参与混合）；调用方负责复制后再交给 Worker。 */
  channels: readonly Float32Array[];
}

export interface RepeatSearchRequest extends RepeatSearchSource {
  id?: number;
  /** 样本章节起止（秒），会被钳制到 [0, duration]。 */
  sampleStart: number;
  sampleEnd: number;
}

export interface RepeatHit {
  /** 命中起止时间（秒），长度与样本一致。 */
  start: number;
  end: number;
  /** 相似度（探针平均 Pearson r，0～1）。 */
  score: number;
}

export type RepeatRejectReason =
  | 'no-audio'
  | 'sample-too-short'
  | 'sample-covers-all'
  | 'sample-silent'
  | 'no-hits';

export interface RepeatSearchResult {
  hits: RepeatHit[];
  /** 实际使用的样本区间（钳制后的秒）。 */
  sampleStart: number;
  sampleEnd: number;
  sampleDuration: number;
  /** 未产出命中的原因（hits 为空时给出，供面板区分「无重复」与其他情况）。 */
  reason: RepeatRejectReason | null;
  /** 能量粗筛扫描的候选位置数（信息展示用）。 */
  scannedPositions: number;
}

export interface RepeatSearchResponse extends RepeatSearchResult {
  id: number;
}

/** 由 AudioBuffer 结构取搜索输入（与 peakSourceFromBuffer 同口径，至多双声道）。 */
export function repeatSourceFromBuffer(buffer: AudioBufferLike): RepeatSearchSource {
  const channels: Float32Array[] = [];
  for (let c = 0; c < Math.min(2, buffer.numberOfChannels); c++) {
    channels.push(buffer.getChannelData(c));
  }
  return {
    duration: buffer.duration,
    length: buffer.length,
    sampleRate: buffer.sampleRate,
    channels,
  };
}

/**
 * 多声道混合 + 块平均降采样为单声道波形。
 * 每帧先在至多两个声道上取均值（1ch 直通），再按块平均抗混叠。
 * 源采样率低于目标率时直接逐帧混合（无升采样）。
 */
export function buildDownmixedMono(
  channels: readonly Float32Array[],
  length: number,
  sourceRate: number,
  targetRate: number,
): Float32Array {
  const ratio = sourceRate / targetRate;
  if (ratio <= 1) {
    const out = new Float32Array(length);
    mixInto(channels, out, 0, length, 0, 1);
    return out;
  }
  const outLen = Math.max(1, Math.floor(length / ratio));
  const out = new Float32Array(outLen);
  const block = Math.max(1, Math.round(ratio));
  for (let i = 0; i < outLen; i++) {
    const s0 = Math.floor(i * ratio);
    const s1 = Math.min(length, s0 + block);
    if (s1 <= s0) {
      out[i] = mixSample(channels, s0);
      continue;
    }
    let sum = 0;
    for (let s = s0; s < s1; s++) sum += mixSample(channels, s);
    out[i] = sum / (s1 - s0);
  }
  return out;
}

function mixSample(channels: readonly Float32Array[], frame: number): number {
  let v = 0;
  for (let c = 0; c < channels.length; c++) v += channels[c][frame];
  return channels.length > 1 ? v / channels.length : v;
}

/** 混合 [s0, s1) 帧并按 step 步长写入 out（outIdx 起）。当前未使用，保留备用。 */
function mixInto(
  channels: readonly Float32Array[],
  out: Float32Array,
  outIdx: number,
  count: number,
  s0: number,
  step: number,
): void {
  for (let i = 0; i < count; i++) out[outIdx + i] = mixSample(channels, s0 + i * step);
}

/** 分块 RMS 能量轮廓（分块长度 ENERGY_BLOCK_SEC，最后不足一块的并入前块）。 */
export function blockRmsEnvelope(
  mono: Float32Array,
  rate: number,
  blockSec: number = ENERGY_BLOCK_SEC,
): Float32Array {
  const block = Math.max(1, Math.round(blockSec * rate));
  const count = Math.max(1, Math.floor(mono.length / block));
  const env = new Float32Array(count);
  for (let b = 0; b < count; b++) {
    let sumSq = 0;
    const s0 = b * block;
    const s1 = b === count - 1 ? mono.length : s0 + block;
    for (let s = s0; s < s1; s++) {
      const v = mono[s];
      sumSq += v * v;
    }
    env[b] = Math.sqrt(sumSq / (s1 - s0));
  }
  return env;
}

/**
 * 去均值归一化相关（Pearson r）：两个等长窗口的逐样本相关。
 * 任一侧去均值平方和过小（常值 / 静音）时返回 null，表示该位置不可比。
 */
export function normalizedCorrelation(
  a: ArrayLike<number>,
  a0: number,
  b: ArrayLike<number>,
  b0: number,
  n: number,
): number | null {
  let sa = 0;
  let sb = 0;
  for (let k = 0; k < n; k++) {
    sa += a[a0 + k];
    sb += b[b0 + k];
  }
  const ma = sa / n;
  const mb = sb / n;
  let num = 0;
  let da2 = 0;
  let db2 = 0;
  for (let k = 0; k < n; k++) {
    const da = a[a0 + k] - ma;
    const db = b[b0 + k] - mb;
    num += da * db;
    da2 += da * da;
    db2 += db * db;
  }
  if (da2 < ZERO_VARIANCE_EPS || db2 < ZERO_VARIANCE_EPS) return null;
  return num / Math.sqrt(da2 * db2);
}

/**
 * 跨步读取的去均值归一化相关：a 取 a0 + k*strideA，b 取 b0 + k*strideB。
 * 单遍扫描完成求和，不为每帧分配临时数组（能量 / 粗波形粗筛的热路径）。
 */
export function stridedNormalizedCorrelation(
  a: ArrayLike<number>,
  a0: number,
  strideA: number,
  b: ArrayLike<number>,
  b0: number,
  strideB: number,
  n: number,
): number | null {
  let sa = 0;
  let sb = 0;
  for (let k = 0; k < n; k++) {
    sa += a[a0 + k * strideA];
    sb += b[b0 + k * strideB];
  }
  const ma = sa / n;
  const mb = sb / n;
  let num = 0;
  let da2 = 0;
  let db2 = 0;
  for (let k = 0; k < n; k++) {
    const da = a[a0 + k * strideA] - ma;
    const db = b[b0 + k * strideB] - mb;
    num += da * db;
    da2 += da * da;
    db2 += db * db;
  }
  if (da2 < ZERO_VARIANCE_EPS || db2 < ZERO_VARIANCE_EPS) return null;
  return num / Math.sqrt(da2 * db2);
}

interface ScoredOffset {
  offset: number;
  score: number;
}

/**
 * 非极大值抑制：按分数降序保留代表，丢弃与已保留代表距离 < mergeRadius
 * 的位置；至多保留 limit 个。输入不会被修改。
 */
function nonMaxSuppress(scored: ScoredOffset[], mergeRadius: number, limit: number): ScoredOffset[] {
  const sorted = scored.sort((a, b) => b.score - a.score || a.offset - b.offset);
  const kept: ScoredOffset[] = [];
  for (const cand of sorted) {
    let near = false;
    for (const k of kept) {
      if (Math.abs(cand.offset - k.offset) < mergeRadius) {
        near = true;
        break;
      }
    }
    if (!near) kept.push(cand);
    if (kept.length >= limit) break;
  }
  return kept;
}

/** 探针（样本的头 / 中 / 尾分段），精匹配用。 */
interface Probe {
  /** 探针在样本内的帧偏移（fine 帧）。 */
  sampleOffset: number;
  frames: number;
}

/**
 * 为精匹配选取探针：样本不超过 PROBE_MAX_SEC 时整段一支探针；
 * 更长时取头 / 中 / 尾三段，相邻探针间留出约一个探针长度的间隔。
 */
export function buildProbes(sampleFrames: number, fineRate: number): Probe[] {
  const maxFrames = Math.round(PROBE_MAX_SEC * fineRate);
  if (sampleFrames <= maxFrames) {
    return [{ sampleOffset: 0, frames: sampleFrames }];
  }
  const gap = Math.min(maxFrames, Math.floor((sampleFrames - MAX_PROBES * maxFrames) / (MAX_PROBES - 1)));
  const probes: Probe[] = [];
  for (let i = 0; i < MAX_PROBES; i++) {
    const offset = i * (maxFrames + Math.max(0, gap));
    if (offset + maxFrames <= sampleFrames) {
      probes.push({ sampleOffset: offset, frames: maxFrames });
    }
  }
  // 尾探针贴齐样本结尾（保证真正覆盖尾部）
  probes[probes.length - 1] = {
    sampleOffset: sampleFrames - maxFrames,
    frames: maxFrames,
  };
  return probes;
}

/** 区间是否与样本区间重叠（端点相接不算重叠）。 */
function overlapsSample(start: number, end: number, sampleStart: number, sampleEnd: number): boolean {
  return start < sampleEnd - 1e-9 && end > sampleStart + 1e-9;
}

/**
 * 在整段录音中定位样本章节的重复录制位置。
 * @throws Error 当输入为空 / 样本过短 / 覆盖整段 / 全静音（消息供 UI 展示）。
 */
export function findRepeats(req: RepeatSearchRequest): RepeatSearchResult {
  const { duration, length, sampleRate, channels, sampleStart: rawStart, sampleEnd: rawEnd } = req;
  if (!channels.length || length <= 0 || sampleRate <= 0 || duration <= 0) {
    throw new Error('没有可搜索的音频数据。');
  }

  const sampleStart = Math.min(Math.max(rawStart, 0), duration);
  const sampleEnd = Math.min(Math.max(rawEnd, 0), duration);
  const sampleDuration = sampleEnd - sampleStart;
  const empty = (reason: RepeatRejectReason, scanned = 0): RepeatSearchResult => ({
    hits: [],
    sampleStart,
    sampleEnd,
    sampleDuration,
    reason,
    scannedPositions: scanned,
  });

  if (!Number.isFinite(sampleStart) || !Number.isFinite(sampleEnd) || sampleDuration < MIN_SAMPLE_SEC - 1e-9) {
    throw new Error(`样本过短：章节至少需要 ${MIN_SAMPLE_SEC} 秒音频才可用于定位。`);
  }
  // 样本覆盖（或几乎覆盖）整段录音时，没有任何可放置候选的位置
  if (sampleStart <= 1e-6 && duration - sampleEnd <= 1 / FINE_RATE) {
    throw new Error('样本章节几乎覆盖整段录音，没有可搜索的其他区间。');
  }

  // ---- 降采样混合波形（精匹配 + 粗波形共用）----
  const fine = buildDownmixedMono(channels, length, sampleRate, FINE_RATE);
  const fineSampleStart = Math.round(sampleStart * FINE_RATE);
  const fineSampleEnd = Math.min(fine.length, Math.round(sampleEnd * FINE_RATE));
  const sampleFrames = fineSampleEnd - fineSampleStart;
  if (sampleFrames < Math.ceil(MIN_SAMPLE_SEC * FINE_RATE)) {
    throw new Error(`样本过短：章节至少需要 ${MIN_SAMPLE_SEC} 秒音频才可用于定位。`);
  }

  // 全静音样本：探针 RMS 全部低于门限，无法做有意义的匹配
  const probes = buildProbes(sampleFrames, FINE_RATE);
  const liveProbes = probes.filter((p) => {
    let sumSq = 0;
    for (let i = 0; i < p.frames; i++) {
      const v = fine[fineSampleStart + p.sampleOffset + i];
      sumSq += v * v;
    }
    return Math.sqrt(sumSq / p.frames) >= PROBE_SILENCE_RMS;
  });
  if (liveProbes.length === 0) {
    return empty('sample-silent');
  }

  // ---- 第 1 级：能量轮廓归一化相关粗筛（50ms 分块）----
  const energy = blockRmsEnvelope(fine, FINE_RATE, ENERGY_BLOCK_SEC);
  const energyRate = 1 / ENERGY_BLOCK_SEC;
  // 样本起点对齐到能量网格（误差 ≤ 半块，粗筛可接受）
  const sampleBlockStart = Math.round(sampleStart / ENERGY_BLOCK_SEC);
  const sampleBlocks = Math.max(2, Math.round(sampleDuration / ENERGY_BLOCK_SEC));
  const lastEnergyStart = energy.length - sampleBlocks;
  const sampleEnergyStart = Math.min(sampleBlockStart, lastEnergyStart);
  // 样本自身能量是否恒定（常值能量与任何位置相关都无判别力）→ 跳过能量级
  let sampleEnergyVar = 0;
  {
    let m = 0;
    for (let k = 0; k < sampleBlocks; k++) m += energy[sampleEnergyStart + k];
    m /= sampleBlocks;
    for (let k = 0; k < sampleBlocks; k++) {
      const d = energy[sampleEnergyStart + k] - m;
      sampleEnergyVar += d * d;
    }
  }
  const energyUsable = sampleEnergyVar / sampleBlocks >= ZERO_VARIANCE_EPS;

  const energyScored: ScoredOffset[] = [];
  let scannedPositions = 0;
  if (energyUsable) {
    for (let eb = 0; eb <= lastEnergyStart; eb++) {
      const candStart = eb * ENERGY_BLOCK_SEC;
      const candEnd = candStart + sampleDuration;
      if (overlapsSample(candStart, candEnd, sampleStart, sampleEnd)) continue;
      scannedPositions++;
      const r = stridedNormalizedCorrelation(
        energy, sampleEnergyStart, 1,
        energy, eb, 1,
        sampleBlocks,
      );
      if (r !== null && r >= ENERGY_CUTOFF) {
        energyScored.push({ offset: eb, score: r });
      }
    }
  }

  // ---- 第 2 级：200Hz 混合波形归一化相关（候选区域，步长 5ms）----
  const waveRatio = FINE_RATE / COARSE_WAVE_RATE; // 20
  const coarseSampleFrames = Math.max(2, Math.round(sampleDuration * COARSE_WAVE_RATE));
  const coarseTotal = Math.max(1, Math.floor(fine.length / waveRatio));
  const lastCoarseStart = coarseTotal - coarseSampleFrames;
  const coarseSampleStartFrame = Math.min(
    Math.round(sampleStart * COARSE_WAVE_RATE),
    Math.max(0, lastCoarseStart),
  );

  /** 对给定粗波形起点（粗帧）打分；超出范围 / 与样本重叠 / 常值返回 null。 */
  const scoreCoarse = (cb: number): number | null => {
    if (cb < 0 || cb > lastCoarseStart) return null;
    const candStart = cb / COARSE_WAVE_RATE;
    if (overlapsSample(candStart, candStart + sampleDuration, sampleStart, sampleEnd)) return null;
    return stridedNormalizedCorrelation(
      fine, coarseSampleStartFrame * waveRatio, waveRatio,
      fine, cb * waveRatio, waveRatio,
      coarseSampleFrames,
    );
  };

  let coarseSeeds: ScoredOffset[] = [];
  if (energyUsable) {
    // 能量候选（能量块序号）→ 对应粗波形起点，交给第 2 级
    const energyKept = nonMaxSuppress(
      energyScored,
      Math.round(0.2 * energyRate), // 同一处 ±0.1s 只保留最高分
      ENERGY_MAX_CANDIDATES,
    );
    // 每个能量候选向前后各扩 1 个能量块（±50ms），用粗波形相关重新对齐
    const seen = new Set<number>();
    const waveScored: ScoredOffset[] = [];
    for (const ec of energyKept) {
      const center = Math.round((ec.offset * ENERGY_BLOCK_SEC) * COARSE_WAVE_RATE);
      for (let cb = center - 10; cb <= center + 10; cb++) {
        if (cb < 0 || cb > lastCoarseStart || seen.has(cb)) continue;
        seen.add(cb);
        const r = scoreCoarse(cb);
        if (r !== null && r >= COARSE_CUTOFF) waveScored.push({ offset: cb, score: r });
      }
    }
    coarseSeeds = nonMaxSuppress(
      waveScored,
      Math.round(0.1 * COARSE_WAVE_RATE), // ±50ms 内同一峰只留一个
      COARSE_MAX_CANDIDATES,
    );
  } else {
    // 能量无判别力：全范围粗波形扫描（长音频时这一级本身也比满分辨率便宜 20 倍）
    const waveScored: ScoredOffset[] = [];
    for (let cb = 0; cb <= lastCoarseStart; cb++) {
      const r = scoreCoarse(cb);
      if (r !== null) {
        scannedPositions++;
        if (r >= COARSE_CUTOFF) waveScored.push({ offset: cb, score: r });
      }
    }
    coarseSeeds = nonMaxSuppress(
      waveScored,
      Math.round(0.1 * COARSE_WAVE_RATE),
      COARSE_MAX_CANDIDATES,
    );
  }

  // ---- 第 3 级：4kHz 探针归一化相关精匹配（±10ms 邻域取最优对齐）----
  const refineRadius = Math.max(1, Math.round(0.01 * FINE_RATE)); // ±10ms
  const fineHits: RepeatHit[] = [];
  for (const seed of coarseSeeds) {
    const centerFine = Math.round((seed.offset / COARSE_WAVE_RATE) * FINE_RATE);
    let best: { offset: number; score: number } | null = null;
    for (let d = -refineRadius; d <= refineRadius; d++) {
      const candOffset = centerFine + d;
      if (candOffset < 0 || candOffset + sampleFrames > fine.length) continue;
      let sum = 0;
      let used = 0;
      for (const p of liveProbes) {
        const r = normalizedCorrelation(
          fine, fineSampleStart + p.sampleOffset,
          fine, candOffset + p.sampleOffset,
          p.frames,
        );
        if (r !== null) {
          sum += r;
          used++;
        }
      }
      if (used < Math.ceil(liveProbes.length / 2)) continue;
      const mean = sum / used;
      if (!best || mean > best.score) best = { offset: candOffset, score: mean };
    }
    if (!best || best.score < FINE_CUTOFF) continue;
    const start = best.offset / FINE_RATE;
    const end = (best.offset + sampleFrames) / FINE_RATE;
    if (overlapsSample(start, end, sampleStart, sampleEnd)) continue;
    fineHits.push({ start, end, score: best.score });
  }

  // ---- 合并同一片重复段附近的命中（间隔 < 样本长度一半即视为同一处）----
  const mergeGap = Math.max(0.25, sampleDuration / 2);
  const sortedHits = fineHits.sort((a, b) => b.score - a.score);
  const merged: RepeatHit[] = [];
  for (const h of sortedHits) {
    const owner = merged.find(
      (m) => Math.abs(m.start - h.start) < mergeGap || (h.start < m.end && h.end > m.start),
    );
    if (owner) {
      // 已有代表：把边界向新命中扩一点（同一片重复段的对齐抖动），保留最高分
      owner.score = Math.max(owner.score, h.score);
      owner.start = Math.min(owner.start, h.start);
      owner.end = Math.max(owner.end, h.end);
    } else {
      merged.push({ ...h });
    }
  }

  merged.sort((a, b) => b.score - a.score || a.start - b.start);
  const hits = merged.slice(0, MAX_HITS).sort((a, b) => a.start - b.start);
  return {
    hits,
    sampleStart,
    sampleEnd,
    sampleDuration,
    reason: hits.length === 0 ? 'no-hits' : null,
    scannedPositions,
  };
}
