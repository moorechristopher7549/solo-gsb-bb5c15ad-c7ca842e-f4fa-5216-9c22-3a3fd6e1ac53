import {
  findRepeats,
  repeatSourceFromBuffer,
  type RepeatSearchResponse,
  type RepeatSearchResult,
} from './repeatSearch';
import RepeatWorker from './repeatSearch.worker.ts?worker';
import type { AudioBufferLike } from '../types';

/** 开始新搜索时，旧搜索被此错误拒绝（调用方静默忽略即可）。 */
export class RepeatSupersededError extends Error {
  constructor() {
    super('已发起新的重复定位搜索，旧结果作废。');
    this.name = 'RepeatSupersededError';
  }
}

interface PendingJob {
  id: number;
  resolve: (result: RepeatSearchResult) => void;
  reject: (err: unknown) => void;
}

/**
 * 重复录制定位 Worker 客户端（与 PeaksWorkerClient 同一模式）：
 *
 * - 三级匹配全程在 Worker 中完成，主线程只负责发起与展示；
 * - 连续搜索（不断修改样本章节边界后重搜）：发起新任务即终止旧 Worker，
 *   旧任务的 CPU 立刻停掉、promise 以 RepeatSupersededError 拒绝，只有
 *   最新一次搜索能回写结果——界面上永远只展示最新结果；
 * - 只 post 至多两个声道的副本（slice 分配新 ArrayBuffer，可 transfer），
 *   主线程的已解码 AudioBuffer 不被改动；
 * - 不支持 Web Worker 的环境回退到主线程纯函数，功能仍可用。
 */
export class RepeatSearchClient {
  private worker: Worker | null = null;
  private job: PendingJob | null = null;
  private nextId = 1;
  private readonly workersSupported = typeof Worker !== 'undefined';

  search(
    buffer: AudioBufferLike,
    sampleStart: number,
    sampleEnd: number,
  ): Promise<RepeatSearchResult> {
    // 新搜索永远取代旧搜索：先杀掉在跑的任务与 Worker
    this.cancel(new RepeatSupersededError());

    if (!this.workersSupported) {
      return Promise.resolve(
        findRepeats({ ...repeatSourceFromBuffer(buffer), sampleStart, sampleEnd }),
      );
    }

    return new Promise<RepeatSearchResult>((resolve, reject) => {
      const id = this.nextId++;
      this.job = { id, resolve, reject };
      try {
        // 每个活跃任务一个新 Worker：terminate 旧 Worker 保证过期搜索立即停算
        const worker = new RepeatWorker();
        this.worker = worker;
        worker.onmessage = (e: MessageEvent<RepeatSearchResponse | { id: number; error: string }>) => {
          const job = this.job;
          if (!job || job.id !== id) return; // 过期 / 已被取代的回复
          this.job = null;
          worker.terminate();
          if (this.worker === worker) this.worker = null;
          const data = e.data;
          if ('error' in data) {
            job.reject(new Error(data.error));
          } else {
            job.resolve(stripId(data));
          }
        };
        worker.onerror = (e: ErrorEvent) => {
          const job = this.job;
          if (!job || job.id !== id) return;
          this.job = null;
          worker.terminate();
          if (this.worker === worker) this.worker = null;
          job.reject(new Error(e.message || '重复定位 Worker 发生错误。'));
        };

        const source = repeatSourceFromBuffer(buffer);
        // 副本 + transfer：AudioBuffer 自身的声道数据保持不动
        const copies = source.channels.map((data) => data.slice());
        worker.postMessage(
          { id, ...source, channels: copies, sampleStart, sampleEnd },
          copies.map((data) => data.buffer),
        );
      } catch (err) {
        if (this.job?.id === id) {
          this.job = null;
          this.terminateWorker();
        }
        reject(err instanceof Error ? err : new Error('无法启动重复定位 Worker。'));
      }
    });
  }

  /** 取消在跑的搜索（更换音频 / 修改样本边界时调用）。 */
  cancel(reason: unknown = new RepeatSupersededError()): void {
    const job = this.job;
    this.job = null;
    this.terminateWorker();
    job?.reject(reason);
  }

  private terminateWorker(): void {
    const worker = this.worker;
    this.worker = null;
    worker?.terminate();
  }

  dispose(): void {
    this.cancel(new RepeatSupersededError());
  }
}

function stripId(res: RepeatSearchResponse): RepeatSearchResult {
  return {
    hits: res.hits,
    sampleStart: res.sampleStart,
    sampleEnd: res.sampleEnd,
    sampleDuration: res.sampleDuration,
    reason: res.reason,
    scannedPositions: res.scannedPositions,
  };
}
