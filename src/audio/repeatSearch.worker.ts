import {
  findRepeats,
  type RepeatSearchRequest,
  type RepeatSearchResponse,
} from './repeatSearch';

/**
 * 重复录制定位 Worker：三级（能量 → 降采样混合波形 → 归一化相关精匹配）
 * 搜索全部在这里运行，长录音不阻塞主线程的播放与章节编辑。
 *
 * 与峰值 Worker 同口径：主线程只把所需声道（至多两个）的副本 post 过来，
 * 已解码 AudioBuffer 始终留在主线程用于播放。客户端按「一次搜索一个
 * Worker」管理，新搜索开始时旧 Worker 直接 terminate，旧任务立即停算。
 *
 * DOM 与 webworker 的 TS lib 不能在同一 program 混用，这里只用结构类型
 * 描述实际用到的全局作用域。
 */
type RepeatWorkerOut = RepeatSearchResponse | { id: number; error: string };

interface RepeatWorkerScope {
  onmessage: ((ev: MessageEvent<RepeatSearchRequest>) => void) | null;
  postMessage(message: RepeatWorkerOut, transfer: Transferable[]): void;
  postMessage(message: RepeatWorkerOut): void;
}

const worker = self as unknown as RepeatWorkerScope;

worker.onmessage = (e: MessageEvent<RepeatSearchRequest>) => {
  const req = e.data;
  const id = req.id ?? 0;
  try {
    worker.postMessage({ id, ...findRepeats(req) });
  } catch (err) {
    worker.postMessage({
      id,
      error: err instanceof Error ? err.message : '重复录制定位失败。',
    });
  }
};

export {};
