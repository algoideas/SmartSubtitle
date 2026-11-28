import { audio } from '@kit.AudioKit';
import abilityAccessCtrl from '@ohos.abilityAccessCtrl';
import { hilog } from '@kit.PerformanceAnalysisKit';
import worker, { MessageEvents } from '@ohos.worker';
import { fileIo as fs } from '@kit.CoreFileKit';
import { buffer } from '@kit.ArkTS';
import { router } from '@kit.ArkUI';

const TAG = 'SmartSubtitle';
const SAMPLE_RATE = 16000;
// 模型预置在 HAP 的 rawfile/models 下, 运行时复制到沙箱 filesDir 后由
// Worker 线程加载(ONNX Runtime 按路径加载)
const RAWFILE_MODEL_DIR = 'models';

class Caption {
  text: string = '';
  isFinal: boolean = false;
  time: string = '';
}

@Entry
@Component
struct Index {
  @State running: boolean = false;
  @State backend: string = 'cpu';       // "cpu" | "npu"
  @State language: string = 'zh';       // "zh" | "en" (UI hint; model covers zh+en)
  @State captions: Caption[] = [];
  @State lastError: string = '';
  @State buttonPressed: boolean = false;
  @State envChecking: boolean = false;
  // 断点判定(HarmonyOS 规范: md >= 600vp): 宽屏时设置卡片与字幕区双栏并排
  @State isWide: boolean = false;
  // 页面左右留白随断点分级: sm 16vp / md 24vp / lg 48vp
  @State pagePadding: number = 16;

  // 当前推理后端的显示文本(与 backend 状态联动)
  private getBackendLabel(): string {
    return this.backend === 'cpu' ? 'CPU (sherpa-onnx)' : 'NPU (RK3588)';
  }

  private audioCapturer: audio.AudioCapturer | null = null;
  private reading: boolean = false;
  private subtitleWorker: worker.ThreadWorker | null = null;

  onSubtitleUpdate(text: string, isFinal: boolean): void {
    if (text.length === 0) {
      return;
    }
    const now = new Date();
    const stamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    const list = this.captions;
    if (!isFinal && list.length > 0 && !list[list.length - 1].isFinal) {
      // update the in-flight partial caption
      list[list.length - 1].text = text;
      list[list.length - 1].time = stamp;
    } else {
      list.push({ text: text, isFinal: isFinal, time: stamp });
      // keep the visible history bounded
      if (list.length > 200) {
        list.splice(0, list.length - 200);
      }
    }
    this.captions = list.slice();
  }

  async requestMicPermission(): Promise<boolean> {
    try {
      const atManager = abilityAccessCtrl.createAtManager();
      const result = await atManager.requestPermissionsFromUser(
        getContext(this), ['ohos.permission.MICROPHONE']);
      return result.permissions.length > 0 &&
        result.authResults[0] === 0;
    } catch (e) {
      hilog.error(0x0000, TAG, 'permission request failed: %{public}s', JSON.stringify(e));
      return false;
    }
  }

  // 模型预置: 把 rawfile/models 下的模型文件复制到沙箱 filesDir/models
  // (ONNX Runtime 按文件路径加载), 已存在且大小一致则跳过
  private copyModelsToSandbox(): string {
    const context = getContext(this);
    const sandboxBase = context.getApplicationContext().filesDir;
    const modelDir = `${sandboxBase}/${RAWFILE_MODEL_DIR}`;
    if (!fs.accessSync(modelDir)) {
      fs.mkdirSync(modelDir, true);
    }
    const names = ['encoder.int8.onnx', 'decoder.int8.onnx', 'tokens.txt', 'silero_vad.onnx', 'paraformer-encoder.rknn', 'cif.onnx'];
    for (const name of names) {
      const raw = `${RAWFILE_MODEL_DIR}/${name}`;
      const dst = `${modelDir}/${name}`;
      const data = context.resourceManager.getRawFileContentSync(raw);
      let exists = false;
      if (fs.accessSync(dst)) {
        const stat = fs.statSync(dst);
        exists = stat.size === data.byteLength;
      }
      if (!exists) {
        const fp = fs.openSync(dst, fs.OpenMode.WRITE_ONLY | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
        fs.writeSync(fp.fd, buffer.from(data).buffer);
        fs.closeSync(fp.fd);
      }
    }
    return modelDir;
  }

  // 环境检查用预制音频: rawfile/0.wav 复制到沙箱 filesDir(已存在且
  // 大小一致则跳过)
  private copyWavToSandbox(): string {
    const context = getContext(this);
    const dst = `${context.getApplicationContext().filesDir}/0.wav`;
    const data = context.resourceManager.getRawFileContentSync('0.wav');
    let exists = false;
    if (fs.accessSync(dst)) {
      const stat = fs.statSync(dst);
      exists = stat.size === data.byteLength;
    }
    if (!exists) {
      const fp = fs.openSync(dst, fs.OpenMode.WRITE_ONLY | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
      fs.writeSync(fp.fd, buffer.from(data).buffer);
      fs.closeSync(fp.fd);
    }
    return dst;
  }

  // 创建 Worker 并注册公共消息处理(识别结果/引擎状态)
  private createWorker(): worker.ThreadWorker {
    const w = new worker.ThreadWorker('entry/ets/workers/SubtitleWorker.ets');
    w.onmessage = (e: MessageEvents) => {
      const msg = e.data as Record<string, string | boolean>;
      if (msg.type === 'result') {
        this.onSubtitleUpdate(msg.text as string, msg.isFinal as boolean);
      } else if (msg.type === 'started' && !(msg.ok as boolean)) {
        this.lastError = '引擎初始化失败: ' + (msg.error as string);
      } else if (msg.type === 'envcheck-done') {
        this.envChecking = false;
        if (!(msg.ok as boolean)) {
          this.lastError = '环境检查失败: ' + (msg.error as string);
        }
        w.terminate();
        if (this.subtitleWorker === w) {
          this.subtitleWorker = null;
        }
      }
    };
    w.onerror = (e) => {
      this.lastError = 'Worker 异常: ' + e.message;
      this.envChecking = false;
    };
    return w;
  }

  // 环境检查: 复制预制 0.wav 到沙箱, 经 Worker 喂入引擎识别,
  // 结果实时显示在字幕区; 完成后自动停止
  async envCheck(): Promise<void> {
    if (this.running || this.envChecking) {
      return;
    }
    const modelDir = this.copyModelsToSandbox();
    const wavPath = this.copyWavToSandbox();

    this.subtitleWorker = this.createWorker();
    this.envChecking = true;
    this.lastError = '';
    this.captions = [];
    this.subtitleWorker.postMessage({
      type: 'envcheck',
      backend: this.backend,
      modelDir: modelDir,
      wavPath: wavPath
    });
  }

  async start(): Promise<void> {
    if (!(await this.requestMicPermission())) {
      this.lastError = '麦克风权限被拒绝';
      return;
    }

    // 模型预置加载
    const modelDir = this.copyModelsToSandbox();

    // 耗时的原生引擎初始化与推理都在 Worker 线程执行
    this.subtitleWorker = this.createWorker();
    this.subtitleWorker.postMessage({
      type: 'init',
      backend: this.backend,
      modelDir: modelDir
    });

    try {
      await this.startCapture();
      this.running = true;
      this.lastError = '';
    } catch (e) {
      this.subtitleWorker.postMessage({ type: 'stop' });
      this.subtitleWorker.terminate();
      this.subtitleWorker = null;
      this.lastError = '启动音频采集失败: ' + JSON.stringify(e);
    }
  }

  async startCapture(): Promise<void> {
    const streamInfo: audio.AudioStreamInfo = {
      samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
      channels: audio.AudioChannel.CHANNEL_1,
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
      encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
    };
    const capturerInfo: audio.AudioCapturerInfo = {
      source: audio.SourceType.SOURCE_TYPE_MIC,
      capturerFlags: 0
    };
    const options: audio.AudioCapturerOptions = {
      streamInfo: streamInfo,
      capturerInfo: capturerInfo
    };

    this.audioCapturer = await audio.createAudioCapturer(options);
    this.reading = true;
    this.audioCapturer.on('readData', (pcm: ArrayBuffer) => {
      if (this.reading && pcm.byteLength > 0 && this.subtitleWorker) {
        // 采集缓冲为外部内存, Worker 序列化不支持直接克隆, 先拷贝再发送
        this.subtitleWorker.postMessage({ type: 'pcm', buffer: pcm.slice(0) });
      }
    });
    await this.audioCapturer.start();
  }

  async stop(): Promise<void> {
    this.reading = false;
    try {
      if (this.audioCapturer) {
        await this.audioCapturer.stop();
        this.audioCapturer.release();
        this.audioCapturer = null;
      }
    } finally {
      if (this.subtitleWorker) {
        this.subtitleWorker.postMessage({ type: 'stop' });
        this.subtitleWorker.terminate();
        this.subtitleWorker = null;
      }
      this.running = false;
    }
  }

  // 设置卡片: 推理后端 / 字幕语言 (同一卡片, 控件等高)
  @Builder
  settingsCard() {
    Column() {
      Row() {
        Text('推理后端')
          .fontSize(14)
          .fontColor('#A0A0B0')
        Blank()
        // Select 不支持文字对齐, 用居中 Text + 下拉菜单实现
        Text(this.backend === 'cpu' ? 'CPU (sherpa-onnx)' : 'NPU (RK3588)')
          .fontSize(14)
          .fontColor('#E6F2FF')
          .textAlign(TextAlign.Center)
          .backgroundColor('#26263A')
          .borderRadius(8)
          .height(40)
          .width('55%')
          .bindMenu([
            {
              value: 'CPU (sherpa-onnx)',
              action: () => {
                if (!this.running) {
                  this.backend = 'cpu';
                }
              }
            },
            {
              value: 'NPU (RK3588)',
              action: () => {
                if (!this.running) {
                  this.backend = 'npu';
                }
              }
            }
          ])
      }
      .width('100%')
      .height(48)
      .alignItems(VerticalAlign.Center)

      Row() {
        Text('字幕语言')
          .fontSize(14)
          .fontColor('#A0A0B0')
        Blank()
        // Paraformer 中英双语模型同时识别中文与 English, 无需切换
        Text('中文 + English')
          .fontSize(14)
          .fontColor('#E6F2FF')
          .textAlign(TextAlign.Center)
          .height(40)
          .width('55%')
      }
      .width('100%')
      .height(48)
      .alignItems(VerticalAlign.Center)
    }
    .width('100%')
    .constraintSize({ maxWidth: 640 })
    .padding({ left: 16, right: 16, top: 12, bottom: 12 })
    .backgroundColor('#1A1A2E')
    .borderRadius(16)
    .margin({ bottom: 12 })
  }

  // 当前推理后端显示 + 错误提示
  @Builder
  statusTexts() {
    Text(`当前推理后端: ${this.getBackendLabel()}`)
      .fontSize(13)
      .fontColor('#8AB4F8')
      .margin({ bottom: 8 })

    if (this.lastError !== '') {
      Text(this.lastError)
        .fontSize(13)
        .fontColor('#FF6B6B')
        .margin({ bottom: 8 })
    }
  }

  // Subtitle area (卡片化)
  @Builder
  subtitleCard() {
    Column() {
      Scroll() {
        Column() {
          ForEach(this.captions, (item: Caption) => {
            Column() {
              Text(item.time)
                .fontSize(11)
                .fontColor('#8888A0')
              Text(item.text)
                .fontSize(20)
                .fontColor(item.isFinal ? '#FFFFFF' : '#BFC8D8')
                .textAlign(TextAlign.Center)
            }
            .width('94%')
            .padding(10)
            .margin({ bottom: 8 })
            .backgroundColor(item.isFinal ? '#CC101018' : '#80101018')
            .borderRadius(12)
          }, (item: Caption, idx: number) => idx + '-' + item.text)
          if (this.captions.length === 0) {
            Text(this.running ? '正在聆听…' :
              (this.envChecking ? '环境检查进行中…' : '点击下方按钮开始实时字幕'))
              .fontSize(14)
              .fontColor('#6E6E88')
              .textAlign(TextAlign.Center)
          }
        }
        .width('100%')
        .alignItems(HorizontalAlign.Center)
      }
      .width('100%')
      .align(Alignment.Bottom)
      .scrollBar(BarState.Off)
    }
    .layoutWeight(1)
    .width('100%')
    .constraintSize({ maxWidth: 640 })
    .padding(12)
    .backgroundColor('#1A1A2E')
    .borderRadius(16)
  }

  // 环境检查 + 开始识别按键
  @Builder
  actionButtons() {
    // 环境检查: 位于开始识别上方, 识别预制 0.wav 验证端到端链路
    Button(this.envChecking ? '检查中…' : '环境检查')
      .width('100%')
      .constraintSize({ maxWidth: 400 })
      .height(40)
      .fontSize(14)
      .fontColor('#E6F2FF')
      .backgroundColor('#26263A')
      .borderRadius(20)
      .margin({ top: 8 })
      .enabled(!this.running && !this.envChecking)
      .onClick(() => {
        this.envCheck();
      })

    Button(this.running ? '停止' : '开始识别')
      .width('100%')
      .constraintSize({ maxWidth: 400 })
      .height(48)
      .fontSize(16)
      .fontColor('#FFFFFF')
      .backgroundColor(this.running ? '#C8402F' : '#0A59F7')
      .borderRadius(24)
      .margin({ top: 8, bottom: 24 })
      .onTouch((event: TouchEvent) => {
        // 按压反馈: 按下时降低不透明度, 符合 HarmonyOS 交互规范
        if (event.type === TouchType.Down) {
          this.buttonPressed = true;
        } else if (event.type === TouchType.Up || event.type === TouchType.Cancel) {
          this.buttonPressed = false;
        }
      })
      .opacity(this.buttonPressed ? 0.7 : 1)
      .onClick(() => {
        if (this.running) {
          this.stop();
        } else {
          this.start();
        }
      })
  }

  build() {
    Column() {
      Row() {
        Text('实时 AI 字幕')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .fontColor('#E6F2FF')
          .layoutWeight(1)

        Image($r('app.media.ic_menu'))
          .width(24)
          .height(24)
          .objectFit(ImageFit.Contain)
          .fillColor('#E6F2FF')
          .bindMenu([
            {
              value: '关于',
              action: () => {
                router.pushUrl({ url: 'pages/AboutPage' });
              }
            }
          ])
      }
      .width('100%')
      .padding({ top: 12, bottom: 16 })
      .alignItems(VerticalAlign.Center)

      // 断点适配: 窄屏(sm)单栏; 宽屏(md/lg >=600vp)双栏并排
      if (this.isWide) {
        Row() {
          Column() {
            this.settingsCard()
            this.statusTexts()
          }
          .width('42%')

          Column() {
            this.subtitleCard()
            this.actionButtons()
          }
          .layoutWeight(1)
          .margin({ left: 24 })
        }
        .layoutWeight(1)
        .alignItems(VerticalAlign.Top)
        .constraintSize({ maxWidth: 1280 })
      } else {
        this.settingsCard()
        this.statusTexts()
        this.subtitleCard()
        this.actionButtons()
      }
    }
    .width('100%')
    .height('100%')
    .padding({ left: this.pagePadding, right: this.pagePadding })
    .backgroundColor('#12121E')
    .onAreaChange((_old: Area, area: Area) => {
      // HarmonyOS 断点: md >= 600vp, lg >= 840vp; 留白分级 16/24/32vp
      const w = Number(area.width);
      this.isWide = w >= 600;
      this.pagePadding = w >= 840 ? 48 : (w >= 600 ? 24 : 16);
    })
  }
}
