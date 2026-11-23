import { router } from '@kit.ArkUI';

const APP_VERSION = '1.0.0';

interface FeatureItem {
  label: string;
}

interface InfoRow {
  label: string;
  value: string;
}

@Entry
@Component
struct AboutPage {
  private appName: string = '实时 AI 字幕';
  private version: string = APP_VERSION;
  private license: string = 'Apache License 2.0';

  private features: FeatureItem[] = [
    { label: '实时中英文语音识别, 低延迟流式输出' },
    { label: 'CPU 后端: sherpa-onnx 流式 Paraformer, 任意 aarch64 设备' },
    { label: 'NPU 后端: RKNN 推理, 优先适配 RK3588' },
    { label: 'Silero VAD 智能门控, 静音时段零识别开销' },
    { label: 'Worker 线程承载推理, 界面流畅不卡顿' }
  ];

  private credits: InfoRow[] = [
    { label: 'sherpa-onnx', value: '流式语音识别引擎 (Apache-2.0)' },
    { label: 'Silero VAD', value: '语音活动检测模型 (MIT)' },
    { label: 'Paraformer', value: '中英双语流式识别模型' },
    { label: 'OpenHarmony 社区', value: '系统与开发框架' }
  ];

  build() {
    Column() {
      // 标题栏
      Row() {
        Text('‹')
          .fontSize(28)
          .fontColor('#E6F2FF')
          .width(24)
          .textAlign(TextAlign.Center)
          .onClick(() => {
            router.back();
          })
        Text('关于')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor('#E6F2FF')
          .margin({ left: 12 })
      }
      .width('100%')
      .height(56)
      .padding({ left: 16, right: 16 })
      .alignItems(VerticalAlign.Center)

      Scroll() {
        Column() {
          // 应用标识
          Column() {
            Image($r('app.media.app_icon'))
              .width(72)
              .height(72)
              .borderRadius(16)
              .margin({ bottom: 12 })

            Text(this.appName)
              .fontSize(20)
              .fontWeight(FontWeight.Bold)
              .fontColor('#E6F2FF')
              .margin({ bottom: 4 })

            Text(`版本 ${this.version}`)
              .fontSize(12)
              .fontColor('#A0A0B0')
          }
          .width('100%')
          .padding({ top: 24, bottom: 24 })
          .alignItems(HorizontalAlign.Center)

          Divider()
            .color('#2A2A40')
            .width('88%')

          // 主要功能
          Column() {
            Text('主要功能')
              .fontSize(15)
              .fontWeight(FontWeight.Bold)
              .fontColor('#E6F2FF')
              .margin({ bottom: 12 })
              .alignSelf(ItemAlign.Start)

            ForEach(this.features, (item: FeatureItem) => {
              Row() {
                Text('✓')
                  .fontSize(12)
                  .fontColor('#4CAF7D')
                  .width(24)
                  .textAlign(TextAlign.Center)

                Text(item.label)
                  .fontSize(12)
                  .fontColor('#C8C8D8')
                  .layoutWeight(1)
                  .lineHeight(18)
              }
              .width('100%')
              .padding({ top: 8, bottom: 8 })
              .alignItems(VerticalAlign.Top)
            })
          }
          .width('100%')
          .padding({ left: 16, right: 16, top: 20, bottom: 8 })
          .alignItems(HorizontalAlign.Start)

          Divider()
            .color('#2A2A40')
            .width('88%')

          // 致谢
          Column() {
            Text('开源组件')
              .fontSize(15)
              .fontWeight(FontWeight.Bold)
              .fontColor('#E6F2FF')
              .margin({ bottom: 12 })
              .alignSelf(ItemAlign.Start)

            ForEach(this.credits, (item: InfoRow) => {
              Row() {
                Text(item.label)
                  .fontSize(12)
                  .fontColor('#C8C8D8')
                  .width(120)

                Text(item.value)
                  .fontSize(12)
                  .fontColor('#A0A0B0')
                  .layoutWeight(1)
                  .lineHeight(18)
              }
              .width('100%')
              .padding({ top: 4, bottom: 4 })
              .alignItems(VerticalAlign.Top)
            })
          }
          .width('100%')
          .padding({ left: 16, right: 16, top: 20, bottom: 8 })
          .alignItems(HorizontalAlign.Start)

          Divider()
            .color('#2A2A40')
            .width('88%')

          // 法律声明
          Column() {
            Text('法律声明')
              .fontSize(15)
              .fontWeight(FontWeight.Bold)
              .fontColor('#E6F2FF')
              .margin({ bottom: 12 })
              .alignSelf(ItemAlign.Start)

            Text('本应用遵循 Apache License 2.0 开源协议。应用内使用的第三方组件遵循其各自的开源协议, 详见开源组件列表。麦克风数据仅在本机完成识别, 不会上传至任何服务器。')
              .fontSize(12)
              .fontColor('#A0A0B0')
              .lineHeight(20)
          }
          .width('100%')
          .padding({ left: 16, right: 16, top: 20, bottom: 8 })
          .alignItems(HorizontalAlign.Start)

          Text(`© 2026 SmartSubtitle · ${this.license}`)
            .fontSize(12)
            .fontColor('#6E6E88')
            .margin({ top: 32, bottom: 32 })
        }
        .width('100%')
      }
      .layoutWeight(1)
      .width('100%')
      .scrollBar(BarState.Off)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#0A0A14')
  }
}
