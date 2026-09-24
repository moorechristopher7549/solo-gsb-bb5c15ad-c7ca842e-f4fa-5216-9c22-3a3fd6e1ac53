import type { AudioBufferLike } from '../src/types';
import {
  COARSE_WAVE_RATE,
  ENERGY_BLOCK_SEC,
  FINE_RATE,
  MIN_SAMPLE_SEC,
  PROBE_MAX_SEC,
  blockRmsEnvelope,
  buildDownmixedMono,
  buildProbes,
  findRepeats,
  normalizedCorrelation,
  stridedNormalizedCorrelation,
  type RepeatSearchResult,
} from '../src/audio/repeatSearch';

let failures = 0;
function ok(cond: boolean, msg: string) {
  if (cond) console.log(`ok ${msg}`);
  else {
    failures++;
    console.error(`FAIL ${msg}`);
  }
}
function eq(actual: unknown, expected: unknown, msg: string) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) {
    failures++;
    console.error(`FAIL ${msg}: expected ${e}, got ${a}`);
  } else {
    console.log(`ok ${msg}`);
  }
}
function approx(actual: number, expected: number, eps: number, msg: string) {
  if (Math.abs(actual - expected) <= eps) console.log(`ok ${msg}`);
  else {
    failures++;
    console.error(`FAIL ${msg}: expected ≈${expected}, got ${actual}`);
  }
}
function throws(fn: () => unknown, part: string, msg: string) {
  try {
    fn();
    failures++;
    console.error(`FAIL ${msg}: expected throw containing "${part}"`);
  } catch (err) {
    const text = err instanceof Error ? err.message : String(err);
    if (!text.includes(part)) {
      failures++;
      console.error(`FAIL ${msg}: expected message containing "${part}", got "${text}"`);
    } else console.log(`ok ${msg}`);
  }
}

function makeBuffer(channels: Float32Array[], sampleRate: number): AudioBufferLike {
  return {
    duration: channels[0].length / sampleRate,
    length: channels[0].length,
    sampleRate,
    numberOfChannels: channels.length,
    getChannelData: (ch) => channels[ch],
  };
}

/** 可复现的伪白噪声（LCG）。 */
function whiteNoise(frames: number, seed: number, amp = 0.5): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    s = (1664525 * s + 1013904223) >>> 0;
    out[i] = ((s / 0xffffffff) * 2 - 1) * amp;
  }
  return out;
}

function hitNear(hits: RepeatSearchResult['hits'], t: number, tol = 0.05): boolean {
  return hits.some((h) => Math.abs(h.start - t) <= tol);
}

/** 直接以声道数组发起搜索（测试用，绕过 AudioBuffer 结构）。 */
function search(
  channels: Float32Array[],
  rate: number,
  start: number,
  end: number,
): RepeatSearchResult {
  const b = makeBuffer(channels, rate);
  return findRepeats({
    duration: b.duration,
    length: b.length,
    sampleRate: b.sampleRate,
    channels,
    sampleStart: start,
    sampleEnd: end,
  });
}

// ---- 基础工具 ----
{
  // 双声道混合：(L+R)/2
  const mono = buildDownmixedMono(
    [new Float32Array([0.2, 0.4]), new Float32Array([0, 0.2])],
    2,
    1000,
    1000,
  );
  approx(mono[0], 0.1, 1e-6, 'stereo downmix frame 0');
  approx(mono[1], 0.3, 1e-6, 'stereo downmix frame 1');
  // 块平均降采样：8000→4000，相邻两帧取均值
  const ds = buildDownmixedMono([new Float32Array([0.2, 0.4, 0.8, 0.0])], 4, 8000, FINE_RATE);
  eq(ds.length, 2, 'downsample length');
  approx(ds[0], 0.3, 1e-6, 'downsample block 0 average');
  approx(ds[1], 0.4, 1e-6, 'downsample block 1 average');

  const env = blockRmsEnvelope(new Float32Array(100).fill(0.5), FINE_RATE, ENERGY_BLOCK_SEC);
  // 200 帧/块；100 帧不足一块 → count=1（最后余量并入）
  eq(env.length, 1, 'envelope merges trailing partial block');
  approx(env[0], 0.5, 1e-6, 'envelope rms value');
}

// ---- 归一化相关：对增益 / 直流偏移不敏感 ----
{
  const a = whiteNoise(2000, 7);
  const b = a.map((v) => v * 0.25 + 0.4); // 整体音量变化 + 直流
  approx(normalizedCorrelation(a, 0, b, 0, 2000)!, 1, 1e-6, 'gain/offset invariant correlation');
  const stride = FINE_RATE / COARSE_WAVE_RATE;
  approx(
    stridedNormalizedCorrelation(a, 0, 1, b, 0, 1, 2000)!,
    1,
    1e-6,
    'strided correlation equals dense for stride 1',
  );
  const c = whiteNoise(2000 * stride, 11);
  const d = c.map((v) => v * 0.4);
  approx(
    stridedNormalizedCorrelation(c, 0, stride, d, 0, stride, 2000)!,
    1,
    1e-6,
    'strided gain-invariant correlation',
  );
  const z = new Float32Array(500);
  eq(normalizedCorrelation(z, 0, a, 0, 500), null, 'constant window yields null');
}

// ---- 探针：短样本一支，长样本头中尾三支 ----
{
  const shortP = buildProbes(Math.round(3 * FINE_RATE), FINE_RATE);
  eq(shortP.length, 1, 'short sample: one probe');
  eq(shortP[0].frames, 3 * FINE_RATE, 'short probe covers whole sample');

  const longFrames = Math.round(60 * FINE_RATE);
  const longP = buildProbes(longFrames, FINE_RATE);
  eq(longP.length, 3, 'long sample: three probes');
  eq(longP[0].sampleOffset, 0, 'first probe at head');
  eq(longP[2].sampleOffset + longP[2].frames, longFrames, 'last probe ends at sample tail');
  eq(longP[0].frames, Math.round(PROBE_MAX_SEC * FINE_RATE), 'probe capped at PROBE_MAX_SEC');
}

// ---- 端到端：白噪声录音中找到完全重复的片段 ----
{
  const RATE = 8000;
  const frames = 40 * RATE;
  const noise = whiteNoise(frames, 42);
  const sig = noise.slice();
  // 样本 [10,14)，原样复制到 [28,32)（块对齐于 4kHz 网格）
  sig.set(noise.subarray(10 * RATE, 14 * RATE), 28 * RATE);

  const res = search([sig], RATE, 10, 14);
  ok(res.hits.length >= 1, 'finds at least one repeat');
  ok(hitNear(res.hits, 28), 'repeat located near 28s');
  const hit = res.hits.find((h) => Math.abs(h.start - 28) <= 0.05)!;
  ok(hit.score > 0.95, `exact repeat scores ~1 (got ${hit.score.toFixed(3)})`);
  approx(hit.end - hit.start, 4, 0.02, 'hit length equals sample length');
  ok(!hitNear(res.hits, 10, 0.2), 'sample itself excluded');
  eq(res.reason, null, 'reason null when hits exist');
  approx(res.sampleStart, 10, 1e-6, 'sample start echoed back');
  ok(res.scannedPositions > 0, 'scanned positions reported');
}

// ---- 整体音量变化不影响排序（相关而非差值）----
{
  const RATE = 8000;
  const frames = 40 * RATE;
  const noise = whiteNoise(frames, 99);
  const sig = noise.slice();
  // 以 0.3 倍音量 + 直流偏移复制到 [25,28)
  for (let i = 0; i < 3 * RATE; i++) sig[25 * RATE + i] = noise[5 * RATE + i] * 0.3 + 0.15;
  const res = search([sig], RATE, 5, 8);
  const hit = res.hits.find((h) => Math.abs(h.start - 25) <= 0.05);
  ok(!!hit, 'quieted repeat still found');
  ok(hit !== undefined && hit.score > 0.9, `quieted repeat scores high (got ${hit?.score.toFixed(3)})`);
}

// ---- 与样本重叠的候选必须排除；非重叠处仍可找到 ----
{
  const RATE = 8000;
  const frames = 40 * RATE;
  const noise = whiteNoise(frames, 123);
  const sig = noise.slice();
  // 先放干净副本（在任何对样本的破坏之前）
  sig.set(noise.subarray(10 * RATE, 14 * RATE), 30 * RATE);
  // 再放与样本 [10,14) 仅重叠 0.1s 的副本 [13.9,17.9)（仅破坏样本尾部 0.1s）
  sig.set(noise.subarray(10 * RATE, 14 * RATE), Math.round(13.9 * RATE));
  const res = search([sig], RATE, 10, 14);
  ok(hitNear(res.hits, 30), 'non-overlapping repeat found at 30s');
  ok(
    !res.hits.some((h) => h.start > 10 + 0.1 && h.start < 14 - 0.05),
    'overlapping candidate excluded',
  );
}

// ---- 同一片重复段附近的多个命中合并；远处的保留 ----
{
  const RATE = 8000;
  const frames = 40 * RATE;
  const noise = whiteNoise(frames, 321);
  const sig = noise.slice();
  // 28.0s 与 28.3s（间隔 < 样本长度一半 2s）是同一重复段；35s 是另一段
  sig.set(noise.subarray(10 * RATE, 14 * RATE), Math.round(28 * RATE));
  sig.set(noise.subarray(10 * RATE, 14 * RATE), Math.round(28.3 * RATE));
  sig.set(noise.subarray(10 * RATE, 14 * RATE), 35 * RATE);
  const res = search([sig], RATE, 10, 14);
  const near28 = res.hits.filter((h) => h.start >= 27.5 && h.start <= 28.9);
  eq(near28.length, 1, 'nearby duplicates merged into one hit');
  ok(hitNear(res.hits, 35), 'distant repeat kept separately');
}

// ---- 无重复：纯噪声 ----
{
  const RATE = 8000;
  const sig = whiteNoise(40 * RATE, 555);
  const res = search([sig], RATE, 10, 12);
  eq(res.hits.length, 0, 'no hits for random audio');
  eq(res.reason, 'no-hits', 'reason no-hits');
}

// ---- 全静音样本 ----
{
  const RATE = 8000;
  const sig = new Float32Array(20 * RATE);
  // 背景有一点噪声，但样本章节 [5,8) 全零
  sig.set(whiteNoise(3 * RATE, 8, 0.3), 15 * RATE);
  const res = search([sig], RATE, 5, 8);
  eq(res.hits.length, 0, 'silent sample gives no hits');
  eq(res.reason, 'sample-silent', 'reason sample-silent');
}

// ---- 立体声：混合声道后同样可定位 ----
{
  const RATE = 8000;
  const frames = 30 * RATE;
  const n = whiteNoise(frames, 777);
  const l = n.slice();
  const r = n.slice();
  // 两侧声道同步复制 [6,9) → [22,25)
  l.set(n.subarray(6 * RATE, 9 * RATE), 22 * RATE);
  r.set(n.subarray(6 * RATE, 9 * RATE), 22 * RATE);
  const res = search([l, r], RATE, 6, 9);
  ok(hitNear(res.hits, 22), 'stereo repeat found near 22s');
}

// ---- 非法输入 ----
{
  const RATE = 8000;
  const sig = whiteNoise(10 * RATE, 1);
  throws(
    () => search([sig], RATE, 2, 2 + (MIN_SAMPLE_SEC - 0.2)),
    '至少需要',
    'sample shorter than minimum throws',
  );
  throws(
    () => search([sig], RATE, 0, 10),
    '整段录音',
    'sample covering whole file throws',
  );
  // 越界样本先钳制再处理
  const res = search([sig], RATE, -5, 2);
  approx(res.sampleStart, 0, 1e-6, 'out-of-range sample start clamped');
}

if (failures > 0) {
  console.error(`\n${failures} test(s) FAILED`);
  process.exit(1);
} else {
  console.log('\nall repeat-search tests passed');
}
