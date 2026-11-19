import { audio } from '@kit.AudioKit';
import { permission } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { startEngine, stopEngine, feedAudio } from 'libentry.so';

const TAG = 'SmartSubtitle';
const SAMPLE_RATE = 16000;
const MODEL_DIR = '/data/local/tmp/smartsubtitle/models';

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

  private audioCapturer: audio.AudioCapturer | null = null;
  private reading: boolean = false;

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
      const result = await permission.requestPermissionsFromContext(
        getContext(this), ['ohos.permission.MICROPHONE']);
      return result.permissions.length > 0 &&
        result.authResults[0] === 0;
    } catch (e) {
      hilog.error(0x0000, TAG, 'permission request failed: %{public}s', JSON.stringify(e));
      return false;
    }
  }

  async start(): Promise<void> {
    if (!(await this.requestMicPermission())) {
      this.lastError = '麦克风权限被拒绝';
      return;
    }

    (globalThis as Record<string, Object>)['onSubtitle'] =
      (text: string, isFinal: boolean) => this.onSubtitleUpdate(text, isFinal);

    const ok = startEngine(this.backend, MODEL_DIR);
    if (!ok) {
      this.lastError = '引擎初始化失败，请检查模型目录 ' + MODEL_DIR;
      return;
    }

    try {
      await this.startCapture();
      this.running = true;
      this.lastError = '';
    } catch (e) {
      stopEngine();
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
    this.audioCapturer.on('readData', (buffer: ArrayBuffer) => {
      if (this.reading && buffer.byteLength > 0) {
        feedAudio(buffer);
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
      stopEngine();
      this.running = false;
    }
  }

  build() {
    Column() {
      Text('实时 AI 字幕')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#FFFFFF')
        .margin({ top: 24, bottom: 8 })

      Row() {
        Text('推理后端')
          .fontSize(14)
          .fontColor('#A0A0B0')
        Select([
          { value: 'CPU (sherpa-onnx)' },
          { value: 'NPU (RK3588)' }
        ])
          .selected(!this.running ? (this.backend === 'cpu' ? 0 : 1) : (this.backend === 'cpu' ? 0 : 1))
          .enabled(!this.running)
          .onSelect((index: number) => {
            this.backend = index === 0 ? 'cpu' : 'npu';
          })
          .width('60%')

        Text('字幕语言')
          .fontSize(14)
          .fontColor('#A0A0B0')
          .margin({ left: 16 })
        Select([
          { value: '中文' },
          { value: 'English' }
        ])
          .selected(this.language === 'zh' ? 0 : 1)
          .enabled(!this.running)
          .onSelect((index: number) => {
            this.language = index === 0 ? 'zh' : 'en';
          })
          .width('20%')
      }
      .width('90%')
      .margin({ bottom: 12 })

      if (this.lastError !== '') {
        Text(this.lastError)
          .fontSize(13)
          .fontColor('#FF6B6B')
          .margin({ bottom: 8 })
      }

      // Subtitle area
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
            .width('92%')
            .padding(10)
            .margin({ bottom: 8 })
            .backgroundColor(item.isFinal ? '#CC101018' : '#80101018')
            .borderRadius(12)
          }, (item: Caption, idx: number) => idx + '-' + item.text)
        }
        .width('100%')
        .alignItems(HorizontalAlign.Center)
      }
      .layoutWeight(1)
      .width('100%')
      .align(Alignment.Bottom)

      Button(this.running ? '停止' : '开始识别')
        .width('70%')
        .height(48)
        .backgroundColor(this.running ? '#D03030' : '#3060D0')
        .margin({ top: 8, bottom: 24 })
        .onClick(() => {
          if (this.running) {
            this.stop();
          } else {
            this.start();
          }
        })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#0A0A14')
  }
}
