import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
// 主线程预加载原生模块: 避免 Worker 首次加载 libentry.so 时触发
// musl dlopen 的符号解析崩溃(含 RKNN 链接后尤为必现)
import { getLastError } from 'libentry.so';

const TAG = 'SmartSubtitle';

export default class EntryAbility extends UIAbility {
  onCreate(): void {
    hilog.info(0x0000, TAG, 'EntryAbility onCreate');
  }

  onDestroy(): void {
    hilog.info(0x0000, TAG, 'EntryAbility onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(0x0000, TAG, 'loadContent failed: %{public}d', err.code);
        return;
      }
      hilog.info(0x0000, TAG, 'loadContent done');
    });
  }
}
