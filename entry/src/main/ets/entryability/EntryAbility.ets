import { UIAbility } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
// 主线程预加载原生模块: 避免 Worker 首次加载 libentry.so 时触发
// musl dlopen 的符号解析崩溃
// 注意: 不在主线程预加载 RKNN 运行时本身——部分板卡(如 Orange Pi 5B
// OHOS 6.1)上 dlopen librknnrt.so 在 HAP 进程内必现 musl 崩溃,
// NPU 推理验证走 CLI 路径(见 README)
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
