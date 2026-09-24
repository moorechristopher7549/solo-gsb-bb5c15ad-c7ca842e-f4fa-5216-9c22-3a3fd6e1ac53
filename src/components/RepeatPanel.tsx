import type { Chapter } from '../types';
import {
  MIN_SAMPLE_SEC,
  type RepeatSearchResult,
} from '../audio/repeatSearch';
import { formatTime } from '../utils/time';

interface RepeatPanelProps {
  /** 当前选中章节（样本来源）。 */
  selected: Chapter | null;
  searching: boolean;
  /** 最近一次与当前样本匹配的搜索结果；更换样本 / 音频后由上层清空。 */
  result: RepeatSearchResult | null;
  error: string | null;
  onSearch: () => void;
  onCancel: () => void;
  /** 点击命中：跳到该起点并播放试听。 */
  onPreviewAt: (t: number) => void;
}

function scoreClass(score: number): string {
  if (score >= 0.9) return 'rep-score-high';
  if (score >= 0.75) return 'rep-score-mid';
  return 'rep-score-low';
}

/**
 * 重复录制定位面板：
 * - 以当前选中章节的原始音频为样本，在整段录音中寻找重录位置；
 * - 搜索（能量粗筛 → 降采样混合波形 → 归一化相关精匹配）在 Web Worker
 *   中运行，连续搜索只有最新一次的结果会显示；
 * - 命中按时间排列，展示起止与相似度（对整体音量变化不敏感的 Pearson r），
 *   点击任一条即跳到该起点试听；
 * - 更换音频或修改样本章节边界时，由上层清空旧结果并取消在跑的搜索。
 */
export function RepeatPanel(props: RepeatPanelProps) {
  const { selected, result, searching } = props;
  const sampleDuration = selected ? selected.end - selected.start : 0;
  const tooShort = selected !== null && sampleDuration < MIN_SAMPLE_SEC - 1e-9;
  const canSearch = selected !== null && !tooShort && !searching;

  return (
    <section className="silence-panel repeat-panel">
      <h2>重复录制定位</h2>
      {!selected ? (
        <p className="silence-info">
          先在波形或左侧列表选中一个章节，将以该章节的原始音频为样本，
          在整段录音中寻找重录位置。
        </p>
      ) : (
        <>
          <div className="repeat-sample" title="样本章节：取该章节的原始音频作为搜索样本">
            <span className="repeat-sample-name">样本：{selected.title || '未命名章节'}</span>
            <span className="repeat-sample-time">
              {formatTime(selected.start)} → {formatTime(selected.end)}（{sampleDuration.toFixed(2)}s）
            </span>
          </div>
          {tooShort && (
            <div className="imp-error-text">
              样本章节短于 {MIN_SAMPLE_SEC} 秒，判别力不足，请选择更长的章节或调整边界。
            </div>
          )}
          <div className="repeat-actions">
            <button
              type="button"
              className="primary"
              disabled={!canSearch}
              onClick={props.onSearch}
              title="在整段录音中搜索与样本章节相同（允许整体音量不同）的重复片段"
            >
              🔍 搜索重复位置
            </button>
            {searching && (
              <button type="button" onClick={props.onCancel}>
                取消
              </button>
            )}
          </div>
          {props.error && <div className="imp-error-text">{props.error}</div>}

          {searching && (
            <div className="silence-info">
              正在后台线程比对波形（能量粗筛 → 归一化相关），长录音请稍候…
            </div>
          )}

          {!searching && result && (
            <div className="repeat-result">
              {result.hits.length === 0 ? (
                <div className="silence-info">
                  {result.reason === 'sample-silent'
                    ? '样本章节近似全静音，无法据此定位重复片段。'
                    : '未在整段录音中找到与样本足够相似的重复片段（已自动排除样本自身及重叠区间）。'}
                </div>
              ) : (
                <>
                  <div className="silence-info repeat-summary">
                    找到 {result.hits.length} 处重复候选（已排除样本自身及重叠区间，
                    合并同一重复段附近的命中；相似度按整体音量归一化）
                  </div>
                  <div className="repeat-hit-list">
                    {result.hits.map((h, i) => (
                      <button
                        key={`${h.start.toFixed(3)}-${i}`}
                        type="button"
                        className="repeat-hit"
                        onClick={() => props.onPreviewAt(h.start)}
                        title={`从 ${formatTime(h.start)} 开始播放试听（相似度 ${(h.score * 100).toFixed(1)}%）`}
                      >
                        <span className="repeat-hit-idx">{i + 1}</span>
                        <span className="repeat-hit-time">
                          {formatTime(h.start)} → {formatTime(h.end)}
                        </span>
                        <span className={`repeat-hit-score ${scoreClass(h.score)}`}>
                          {(h.score * 100).toFixed(1)}%
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
