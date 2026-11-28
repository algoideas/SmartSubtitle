// NPU backend: RKNN 上跑 Paraformer 流式编码器, CPU 上跑 CTC 解码器.
//
// 特征管线与 sherpa-onnx 的 online-recognizer-paraformer-impl.h 保持一致:
//   fbank(80) --LFR(7帧堆叠,步长6)--> 560 维 --CMVN--> --位置编码-->
//   左右文缓存(5+3帧)拼成 [1,18,560] --> RKNN 编码器 --> enc [1,18,512]
//   --> CIF 静音阈值搜索 --> 声学嵌入 --> CPU 解码器(ONNX) --> token id
//
// 编解码模型与 tokens 均来自 model_dir (与 CPU 后端共用);
// 编码器使用 paraformer-encoder.rknn (RKNN 转换, 见 third_party/rknn)。
#if defined(SSUB_HAS_RKNN) && defined(SSUB_HAS_SHERPA)

#include <rknn_api.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-paraformer-model.h"
#include "sherpa-onnx/csrc/online-stream.h"

#include "asr_backend.h"
#include "manual_fbank.h"

namespace subtitle {

namespace {

constexpr int kSampleRate = 16000;
// sherpa 流式 Paraformer 的分块参数: 61 帧一窗, 步进 60 帧 (600ms)
constexpr int kChunkFrames = 61;
constexpr int kLeftChunkFrames = 5;
constexpr int kRightChunkFrames = 3;

// y[i] += x[i] * scale
void ScaleAddInPlace(const float *x, int32_t n, float scale, float *y) {
  for (int32_t i = 0; i != n; ++i) {
    y[i] += x[i] * scale;
  }
}

// y[i] = x[i] * scale
void Scale(const float *x, int32_t n, float scale, float *y) {
  for (int32_t i = 0; i != n; ++i) {
    y[i] = x[i] * scale;
  }
}

class NpuBackend : public AsrBackend {
 public:
  explicit NpuBackend(bool use_ort) : use_ort_(use_ort) {}

  bool Init(const BackendConfig& config, AsrCallback callback) override {
    callback_ = std::move(callback);

    // 1) sherpa 模型: 用 int8 编码器读取元数据(CMVN/LFR/分块参数),
    //    解码器 CTC 在 CPU(ONNX) 上推理
    sherpa_onnx::OnlineModelConfig model_cfg;
    model_cfg.model_type = "paraformer";
    model_cfg.paraformer.encoder = config.modelDir + "/encoder.int8.onnx";
    model_cfg.paraformer.decoder = config.modelDir + "/decoder.int8.onnx";
    model_cfg.tokens = config.modelDir + "/tokens.txt";

    model_ = std::make_unique<sherpa_onnx::OnlineParaformerModel>(model_cfg);

    std::ifstream f(config.modelDir + "/tokens.txt");
    if (!f) {
      lastError_ = "failed to load tokens.txt from " + config.modelDir;
      return false;
    }
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) {
        continue;
      }
      auto sp = line.find(' ');
      if (sp == std::string::npos) {
        continue;
      }
      // tokens.txt 格式: "token id" (词在前, id 在后)
      try {
        int32_t id = std::stoi(line.substr(sp + 1));
        tokens_[id] = line.substr(0, sp);
      } catch (const std::exception&) {
        continue;  // 跳过非法行
      }
    }

    // 2) 特征提取: 使用 ManualFbank (与主机已验证可正确解码的参考
    // 管线 tools/host_ref.py 数学一致), 替代 knf OnlineStream
    mf_ = std::make_unique<ManualFbank>();

    // 3) RKNN 编码器 (ORT 模式下走 CPU 编码器, 跳过 RKNN 初始化)
    if (!use_ort_) {
      // 拼接路线 (NPU 浅层 + CPU 深层): 前段 .rknn (前 N 块, int8 保真度高)
      // + 尾部 .onnx (块 N+1..49, ORT), 块 N 残差处 fp32 精确交接。绕开
      // RK3588 深层残差流动态范围超 int8 表征的缺陷 (见 README NPU 章节)。
      // 前段/尾部不齐全时回退全量 paraformer-encoder.rknn。
      const std::string front_path =
          config.modelDir + "/paraformer-encoder-front.rknn";
      const std::string tail_path =
          config.modelDir + "/paraformer-encoder-tail.onnx";
      {
        std::ifstream front_f(front_path);
        std::ifstream tail_f(tail_path);
        use_split_ = front_f.good() && tail_f.good();
      }
      if (use_split_) {
        // 注意: 不使用 GetInputNameAllocated —— CLI 静态链接的 ORT 中该
        // C API 为坏调用 (板端实测 SEGV_MAPERR), 与外置 CIF 路径相同改用
        // 硬编码输入名 (尾部模型输入名由 tools/make_tail_model.py 固定为
        // mid / speech_lengths)
        try {
          Ort::SessionOptions so;
          so.SetIntraOpNumThreads(1);
          tail_sess_ =
              std::make_unique<Ort::Session>(cif_env_, tail_path.c_str(), so);
        } catch (const std::exception& e) {
          use_split_ = false;
          lastError_ = std::string("failed to load tail model: ") + e.what();
          fprintf(stderr, "npu: %s (fallback to full encoder)\n",
                  lastError_.c_str());
        } catch (...) {
          use_split_ = false;
          lastError_ = "failed to load tail model: unknown exception";
          fprintf(stderr, "npu: %s (fallback to full encoder)\n",
                  lastError_.c_str());
        }
      }
      const std::string rknn_path =
          use_split_ ? front_path : config.modelDir + "/paraformer-encoder.rknn";
      if (LoadModel(rknn_path) != RKNN_SUCC) {
        lastError_ = "failed to load " + rknn_path;
        return false;
      }
      // NPU 核掩码: 默认 core0; SSUB_NPU_CORE 覆盖 (0=auto 1/2/4 单核 7=全核)
      static rknn_core_mask core_mask = [] {
        const char* e = getenv("SSUB_NPU_CORE");
        if (e != nullptr) {
          return static_cast<rknn_core_mask>(atoi(e));
        }
        return RKNN_NPU_CORE_0;
      }();
      rknn_set_core_mask(ctx_, core_mask);
      if (getenv("SSUB_NPU_DEBUG")) {
        fprintf(stderr, "npu: core_mask=%d\n", static_cast<int>(core_mask));
      }

      // 查询输入属性: 记录 fmt/size 供 rknn_inputs_set 使用
      rknn_tensor_attr in_attr;
      memset(&in_attr, 0, sizeof(in_attr));
      in_attr.index = 0;
      if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr,
                     sizeof(in_attr)) != RKNN_SUCC) {
        lastError_ = "rknn_query input attr failed";
        return false;
      }
      in_fmt_ = in_attr.fmt;
      in_elems_ = 1;
      for (int i = 0; i < RKNN_MAX_DIMS; i++) {
        if (in_attr.dims[i] > 0) {
          in_elems_ *= in_attr.dims[i];
        }
      }
      if (in_elems_ != 18 * 560) {
        fprintf(stderr, "npu: unexpected encoder input elems %u (expect %d)\n",
                in_elems_, 18 * 560);
      }

      // 输出数量: enc 在 0, alphas 在最后一个 (enc_len 为 int64, 不请求)
      rknn_input_output_num io_num;
      memset(&io_num, 0, sizeof(io_num));
      if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num,
                     sizeof(io_num)) != RKNN_SUCC) {
        lastError_ = "rknn_query output num failed";
        return false;
      }
      alpha_index_ = io_num.n_output - 1;
      if (getenv("SSUB_NPU_DEBUG")) {
        fprintf(stderr, "npu: encoder in=%u out=%u alpha_index=%u\n",
                io_num.n_input, io_num.n_output, alpha_index_);
        for (uint32_t i = 0; i < io_num.n_input; ++i) {
          rknn_tensor_attr a;
          memset(&a, 0, sizeof(a));
          a.index = i;
          if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a)) ==
              RKNN_SUCC) {
            fprintf(stderr, "npu: in[%u] name=%s dims=[", i, a.name);
            for (uint32_t d = 0; d < a.n_dims; ++d)
              fprintf(stderr, "%s%u", d ? "," : "", a.dims[d]);
            fprintf(stderr, "] fmt=%u type=%u size=%u\n",
                    a.fmt, a.type, a.size);
          }
        }
        for (uint32_t i = 0; i < io_num.n_output; ++i) {
          rknn_tensor_attr a;
          memset(&a, 0, sizeof(a));
          a.index = i;
          if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) ==
              RKNN_SUCC) {
            fprintf(stderr, "npu: out[%u] name=%s dims=[", i, a.name);
            for (uint32_t d = 0; d < a.n_dims; ++d)
              fprintf(stderr, "%s%u", d ? "," : "", a.dims[d]);
            fprintf(stderr, "] fmt=%u type=%u size=%u\n",
                    a.fmt, a.type, a.size);
          }
        }
      }

      // 4) 外置 CIF (float32, CPU ORT): RKNN 模型内量化后的 CIF 分支
      //    alphas 保真度差 (板上实测比参考系统性偏高 1.3~5.3 倍, 乱码),
      //    改用 encoder.onnx 提取的 cif.onnx 在 CPU 上计算 (主机逐位验证)。
      //    缺 cif.onnx 时回退 RKNN 内置 alphas (旧行为)。
      {
        const std::string cif_path = config.modelDir + "/cif.onnx";
        std::ifstream cif_f(cif_path);
        if (cif_f.good()) {
          Ort::SessionOptions so;
          so.SetIntraOpNumThreads(1);
          cif_sess_ = std::make_shared<Ort::Session>(cif_env_, cif_path.c_str(), so);
          use_external_cif_ = true;
        }
      }

      // 拼接路线下前段 RKNN 只有 1 个输出 (无 alphas), 必须用外置 CIF
      if (use_split_ && !use_external_cif_) {
        lastError_ = "split route (front+tail) requires cif.onnx in " +
                     config.modelDir;
        return false;
      }
      if (getenv("SSUB_NPU_DEBUG")) {
        fprintf(stderr, "npu: mode=%s\n",
                use_split_ ? "split(front+tail)" : "full");
      }
    }

    feat_dim_ = static_cast<int32_t>(model_->NegativeMean().size());

    running_ = true;
    worker_ = std::thread(&NpuBackend::DecodeLoop, this);
    return true;
  }

  void Feed(const int16_t* pcm, size_t samples) override {
    std::vector<float> floats(samples);
    std::lock_guard<std::mutex> lk(stream_mu_);
    mf_->Feed(pcm, samples);
  }

  void Reset() override {
    std::lock_guard<std::mutex> lk(stream_mu_);
    mf_ = std::make_unique<ManualFbank>();
    num_processed_ = 0;
    feat_cache_.clear();
    initial_hidden_.clear();
    alpha_cache_.assign(1, 0.0f);
    states_.clear();
    utterance_.clear();
    pending_cont_ = false;
  }

  std::string LastError() const override {
    std::lock_guard<std::mutex> lk(stream_mu_);
    return lastError_;
  }

  // 同步排空尾部: 注入静音 + 等待 DecodeLoop 处理完积压帧 + 停止解码线程。
  // 同步语义保证返回时 utterance_ 已完整 (FinalText 可安全读取)。
  // HAP 停止路径 (NAPI StopEngine) 必须先走这里: 末段 partial 若只经
  // threadsafe-function 队列投递, 在 JS 线程释放 tsfn 时它仍阻塞在停止
  // 调用内, 队列中未投递的最终结果会随 tsfn 终结而丢失 (板卡环境检查
  // 实测丢失句尾, 如"是星期三"只剩"是星期")。
  void Finalize() override {
    if (finalized_) {
      return;
    }
    finalized_ = true;
    // 尾部注入 ~0.4s 静音: 音频结束时剩余帧不足 LFR 堆叠窗口(7 帧),
    // 滞留缓存永不编码, 末段 token(如"是星期三")会丢失; 注入静音补齐
    // 窗口使尾部得以编码输出
    {
      std::vector<int16_t> silence(6400, 0);  // 400ms @16k
      for (int i = 0; i < 4; ++i) {
        std::lock_guard<std::mutex> lk(stream_mu_);
        mf_->Feed(silence.data(), silence.size());
      }
    }
    // NPU 解码慢于实时: 停止前等待 DecodeLoop 处理完积压帧(有界),
    // 否则环境检查/短音频场景下结果尚未产出即被丢弃, 字幕为空
    for (int i = 0; i < 300; ++i) {  // 最多 ~30s
      bool drained = false;
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        drained = mf_->AvailableFrames() < num_processed_ + kChunkFrames;
      }
      if (drained) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    running_ = false;
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  std::string FinalText() const override {
    // 仅在 Finalize() 之后调用 (解码线程已 join, 无并发写)
    return utterance_;
  }

  ~NpuBackend() {
    Finalize();
    if (ctx_) {
      rknn_destroy(ctx_);
      ctx_ = 0;
    }
  }

 private:
  int LoadModel(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> model((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (model.empty()) {
      return RKNN_ERR_MODEL_INVALID;
    }
    memset(&ctx_, 0, sizeof(ctx_));
    return rknn_init(&ctx_, model.data(), model.size(), 0, nullptr);
  }

  // ===== 特征处理 (与 sherpa online-recognizer-paraformer-impl.h 一致) =====

  std::vector<float> ApplyLFR(const std::vector<float>& in) const {
    int32_t window = model_->LfrWindowSize();
    int32_t shift = model_->LfrWindowShift();
    int32_t in_dim = 80;
    int32_t in_frames = static_cast<int32_t>(in.size()) / in_dim;
    int32_t out_frames = (in_frames - window) / shift + 1;
    int32_t out_dim = in_dim * window;

    std::vector<float> out(out_frames * out_dim);
    const float* p_in = in.data();
    float* p_out = out.data();
    for (int32_t i = 0; i != out_frames; ++i) {
      std::copy(p_in, p_in + out_dim, p_out);
      p_out += out_dim;
      p_in += shift * in_dim;
    }
    return out;
  }

  void ApplyCMVN(std::vector<float>* v) const {
    const std::vector<float>& neg_mean = model_->NegativeMean();
    const std::vector<float>& inv_stddev = model_->InverseStdDev();
    int32_t dim = static_cast<int32_t>(neg_mean.size());
    int32_t num_frames = static_cast<int32_t>(v->size()) / dim;
    float* p = v->data();
    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        // funasr 导出的 neg_mean 元数据按"待减均值"约定, 正确公式为
        // (v - neg_mean) * inv_stddev (与 vendored 参考
        // online-recognizer-paraformer-impl.h ApplyCMVN 一致)。
        // 误用 (v + neg_mean) 会使特征整体偏移 2*neg_mean, 编码器
        // 输出退化为"首帧质量+末帧 1.0"边界态, CIF 积分永不触发, 无 token。
        p[k] = (p[k] - neg_mean[k]) * inv_stddev[k];
      }
      p += dim;
    }
  }

  void PositionalEncoding(std::vector<float>* v, int32_t t_offset) const {
    int32_t feat_dim = feat_dim_;
    int32_t t_total = static_cast<int32_t>(v->size()) / feat_dim;
    constexpr float kScale = -0.03301197265941284f;

    for (int32_t t = 0; t != t_total; ++t) {
      float* p = v->data() + t * feat_dim;
      int32_t offset = t + 1 + t_offset;
      for (int32_t d = 0; d < feat_dim / 2; ++d) {
        float inv_timescale = offset * std::exp(d * kScale);
        p[d] += std::sin(inv_timescale);
        p[d + feat_dim / 2] += std::cos(inv_timescale);
      }
    }
  }

  // ===== 编码器(RKNN) + CIF + 解码器(ONNX) =====

  void ProcessChunk() {
    std::vector<float> frames;
    int32_t processed_before;
    {
      std::lock_guard<std::mutex> lk(stream_mu_);
      frames = mf_->GetFrames(num_processed_, kChunkFrames);
      if (frames.empty()) {
        return;  // 样本不足一整块, 等待更多音频
      }
      processed_before = num_processed_;
      num_processed_ += kChunkFrames - 1;
    }
    if (getenv("SSUB_NPU_DUMP_RAW")) {
      // 导出原始 fbank 帧 (LFR/CMVN/PE 前), 用于与主机参考对比
      char p[160];
      snprintf(p, sizeof(p), "/data/local/tmp/npu/calib/raw_%03d.bin",
               processed_before);
      FILE* fp = fopen(p, "wb");
      if (fp) {
        fwrite(frames.data(), sizeof(float), frames.size(), fp);
        fclose(fp);
      }
      // 导出 fbank 实际收到的全部 PCM 样本 (int16), 与 wav 原始数据对比
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        FILE* fp2 = fopen("/data/local/tmp/npu/calib/pcm_all.bin", "wb");
        if (fp2) {
          int32_t n = static_cast<int32_t>(mf_->SampleCount());
          std::vector<int16_t> buf(n);
          for (int32_t i = 0; i < n; ++i)
            buf[i] = static_cast<int16_t>(mf_->SampleAt(i));
          fwrite(buf.data(), sizeof(int16_t), buf.size(), fp2);
          fclose(fp2);
        }
      }
    }
    if (getenv("SSUB_NPU_DEBUG")) {
      fprintf(stderr, "npu: ProcessChunk frames=%zu\n", frames.size() / 80);
    }

    frames = ApplyLFR(frames);
    float pre_cmvn_mean0 = 0, pre_cmvn_std0 = 0;
    if (getenv("SSUB_NPU_DEBUG")) {
        // 统计 CMVN 前第 0 维均值/方差, 与模型元数据 neg_mean/inv_stddev 对照
        size_t n_frames = frames.size() / feat_dim_;
        float sum = 0, sq = 0;
        for (size_t i = 0; i < frames.size(); i += feat_dim_) {
            sum += frames[i];
            sq += frames[i] * frames[i];
        }
        pre_cmvn_mean0 = sum / n_frames;
        pre_cmvn_std0 = std::sqrt(sq / n_frames -
                                  pre_cmvn_mean0 * pre_cmvn_mean0);
        fprintf(stderr,
                "npu: pre-CMVN dim0 mean=%.3f std=%.3f (model: -%.3f/%.3f)\n",
                pre_cmvn_mean0, pre_cmvn_std0,
                -model_->NegativeMean()[0], model_->InverseStdDev()[0]);
    }
    // 处理顺序与 sherpa 参考管线
    // (online-recognizer-paraformer-impl.h DecodeStream) 完全一致:
    // 先只对 10 个新 LFR 帧做 CMVN/PE, 再前置 8 帧上下文。
    // 缓存中的上下文帧已带其作为新帧时的 CMVN/PE, 而 PE 是加法叠加,
    // 若拼缓存后再对整窗做 PE, 上下文会被二次叠加 PE 且位置偏移 8 帧,
    // 编码器输出退化为"末帧 alpha=1.0"的边界态, CIF 积分永不触发, 无 token。
    ApplyCMVN(&frames);
    // PE 偏移使用递增前的帧数 (与 sherpa 一致)
    PositionalEncoding(&frames, processed_before / model_->LfrWindowShift());
    // 左右文缓存: 首块补零, 之后与上一块尾部重叠
    if (feat_cache_.empty()) {
      feat_cache_.resize((kLeftChunkFrames + kRightChunkFrames) * feat_dim_, 0);
    }
    frames.insert(frames.begin(), feat_cache_.begin(), feat_cache_.end());
    std::copy(frames.end() - feat_cache_.size(), frames.end(),
              feat_cache_.begin());
    if (getenv("SSUB_NPU_DEBUG")) {
      float s = 0;
      for (float f : frames) {
        if (!std::isfinite(f)) {
          s = NAN;
          break;
        }
      }
      // 帧校验和: 确认各块输入不同
      float sum = 0;
      for (float f : frames) {
        sum += f;
      }
      fprintf(stderr, "npu: after LFR/CMVN/PE dim=%zu nan=%d checksum=%.3f\n",
              frames.size(), static_cast<int>(std::isnan(s)), sum);
    }

    int32_t num_frames = static_cast<int32_t>(frames.size()) / feat_dim_;

    if (getenv("SSUB_NPU_DUMP_CALIB")) {
      static int calib_idx = 0;
      if (calib_idx < 32) {
        char p[160];
        snprintf(p, sizeof(p), "/data/local/tmp/npu/calib/%03d.bin", calib_idx);
        FILE* fp = fopen(p, "wb");
        if (fp) {
          fwrite(frames.data(), sizeof(float), frames.size(), fp);
          fclose(fp);
        }
        calib_idx++;
      }
    }

    // 编码器推理: ORT(CPU 模式) 或 RKNN(NPU 模式)
    std::vector<float> enc_copy;
    std::vector<float> alpha;
    if (use_ort_) {
      // CPU 推理路径: ORT int8 编码器, 与主机参考管线一致
      auto memory_info2 =
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      std::array<int64_t, 3> x_shape{1, num_frames, feat_dim_};
      Ort::Value x = Ort::Value::CreateTensor(
          memory_info2, const_cast<float*>(frames.data()), frames.size(),
          x_shape.data(), 3);
      int32_t x_len = num_frames;
      std::array<int64_t, 1> l_shape{1};
      Ort::Value x_len_v =
          Ort::Value::CreateTensor(memory_info2, &x_len, 1, l_shape.data(), 1);
      auto ref = model_->ForwardEncoder(std::move(x), std::move(x_len_v));
      const float* ref_enc = ref[0].GetTensorData<float>();
      const float* ref_alpha = ref[2].GetTensorData<float>();
      enc_copy.assign(ref_enc, ref_enc + num_frames * 512);
      alpha.assign(ref_alpha, ref_alpha + num_frames);
    } else {
      if (getenv("SSUB_NPU_ZERO_INPUT")) {
        // 诊断: 喂全零特征, 判定 RKNN 输出是否与输入无关
        for (auto& v : frames) v = 0.0f;
      }
      rknn_input in;
      memset(&in, 0, sizeof(in));
      in.index = 0;
      in.type = RKNN_TENSOR_FLOAT32;
      // 数据按 NCHW 布局 [1,18,560] 传入; 若声明为运行时默认的 NHWC,
      // 会被错误重排导致特征 scrambled、编码器输出异常
      in.fmt = RKNN_TENSOR_NCHW;
      in.size = frames.size() * sizeof(float);
      in.buf = frames.data();
      // 非图像/序列模型标准喂法: pass_through=1 跳过运行时预处理(未定义
      // mean/std 时对输入无效), 数据直喂模型。量化模型保留 0, 由运行时做
      // float→int8 量化。SSUB_NPU_PT=1 强制 pass_through。
      in.pass_through = (getenv("SSUB_NPU_PT") != nullptr) ? 1u : 0u;
      if (rknn_inputs_set(ctx_, 1, &in) != RKNN_SUCC) {
        lastError_ = "rknn_inputs_set failed";
        return;
      }
      if (rknn_run(ctx_, nullptr) != RKNN_SUCC) {
        lastError_ = "rknn_run failed";
        fprintf(stderr, "npu: rknn_run failed\n");
        return;
      }

      // 请求 enc(0); 未启用外置 CIF 时再取 alphas(最后); 跳过 int64 的 enc_len
      // (拼接路线前段只有 1 个输出, alphas 一律来自外置 CIF)
      const bool rknn_alpha = !use_external_cif_ && !use_split_;
      const uint32_t n_outs = rknn_alpha ? 2u : 1u;
      rknn_output outs[2];
      memset(outs, 0, sizeof(outs));
      outs[0].index = 0;
      outs[0].want_float = 1;
      outs[0].is_prealloc = 0;
      if (rknn_alpha) {
        outs[1].index = alpha_index_;
        outs[1].want_float = 1;
        outs[1].is_prealloc = 0;
      }
      if (rknn_outputs_get(ctx_, n_outs, outs, nullptr) != RKNN_SUCC) {
        lastError_ = "rknn_outputs_get failed";
        fprintf(stderr, "npu: rknn_outputs_get failed\n");
        return;
      }

      const float* p_enc =
          reinterpret_cast<const float*>(outs[0].buf);  // [1, num_frames, 512]
      const float* p_alpha = rknn_alpha
          ? reinterpret_cast<const float*>(outs[1].buf)  // [1, num_frames]
          : nullptr;
      if (p_alpha != nullptr) {
        alpha.assign(p_alpha, p_alpha + num_frames);
      }

      if (use_split_) {
        // fp32 交接: NPU 前段输出 (块 N 残差) + speech_lengths → ORT 尾部
        // (块 N+1..49 + after_norm + 末端长度相关步骤); 尾部输出即最终
        // hidden states (与全模型 'enc' 同张量), 后续外置 CIF / 解码器不变
        auto mem =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        std::array<int64_t, 3> mid_shape{1, num_frames, 512};
        Ort::Value mid_v = Ort::Value::CreateTensor(
            mem, const_cast<float*>(p_enc), num_frames * 512,
            mid_shape.data(), 3);
        int32_t tail_len = num_frames;
        std::array<int64_t, 1> len_shape{1};
        Ort::Value len_v =
            Ort::Value::CreateTensor(mem, &tail_len, 1, len_shape.data(), 1);
        // 输入名与 tools/make_tail_model.py 的固定命名一致
        const char* tail_in_names[2] = {"mid", "speech_lengths"};
        Ort::Value tail_inputs[2] = {std::move(mid_v), std::move(len_v)};
        const char* tail_out_names[] = {"enc"};
        auto tail_out = tail_sess_->Run(Ort::RunOptions{}, tail_in_names,
                                        tail_inputs, 2, tail_out_names, 1);
        const float* p_tail = tail_out[0].GetTensorData<float>();
        enc_copy.assign(p_tail, p_tail + num_frames * 512);
      } else {
        enc_copy.assign(p_enc, p_enc + num_frames * 512);
      }
      // 拷贝完成后立即释放 RKNN 输出缓冲, 避免悬垂引用
      rknn_outputs_release(ctx_, n_outs, outs);

      if (use_external_cif_) {
        // 外置 CIF: alphas = cif.onnx(enc, speech_lengths), float32
        // (tools/export_cif_subgraph.py, 与编码器内置 alphas 逐位一致)
        auto mem =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        std::array<int64_t, 3> enc_shape{1, num_frames, 512};
        Ort::Value enc_v = Ort::Value::CreateTensor(
            mem, const_cast<float*>(enc_copy.data()), enc_copy.size(),
            enc_shape.data(), 3);
        int32_t len_val = num_frames;
        std::array<int64_t, 1> len_shape{1};
        Ort::Value len_v =
            Ort::Value::CreateTensor(mem, &len_val, 1, len_shape.data(), 1);
        Ort::Value cif_inputs[2] = {std::move(enc_v), std::move(len_v)};
        const char* cif_in_names[] = {"enc", "speech_lengths"};
        const char* cif_out_names[] = {"alphas"};
        auto cif_out = cif_sess_->Run(Ort::RunOptions{}, cif_in_names,
                                      cif_inputs, 2, cif_out_names, 1);
        const float* p_cif = cif_out[0].GetTensorData<float>();
        alpha.assign(p_cif, p_cif + num_frames);
      }
    }
    const float* p_enc = enc_copy.data();
    int32_t enc_dim = 512;
    if (getenv("SSUB_NPU_DEBUG")) {
      float a_sum = 0, a_max = 0;
      for (float a : alpha) {
        a_sum += a;
        a_max = std::max(a_max, a);
      }
      fprintf(stderr, "npu: alpha sum=%.3f max=%.3f\n", a_sum, a_max);
      fprintf(stderr, "npu: alpha vec=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
              "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]\n",
              alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5],
              alpha[6], alpha[7], alpha[8], alpha[9], alpha[10], alpha[11],
              alpha[12], alpha[13], alpha[14], alpha[15], alpha[16],
              alpha[17]);
    }
    std::fill(alpha.begin(), alpha.begin() + kLeftChunkFrames, 0.0f);
    std::fill(alpha.end() - kRightChunkFrames, alpha.end(), 0.0f);

    if (initial_hidden_.empty()) {
      initial_hidden_.resize(enc_dim, 0.0f);
    }
    if (alpha_cache_.empty()) {
      alpha_cache_.assign(1, 0.0f);
    }

    std::vector<float> acoustic_embedding;
    float threshold = 1.0f;
    float integrate = alpha_cache_[0];

    for (int32_t i = 0; i != num_frames; ++i) {
      float this_alpha = alpha[i];
      if (integrate + this_alpha < threshold) {
        integrate += this_alpha;
        ScaleAddInPlace(p_enc + i * enc_dim, enc_dim, this_alpha,
                        initial_hidden_.data());
        continue;
      }
      // fire
      ScaleAddInPlace(p_enc + i * enc_dim, enc_dim, threshold - integrate,
                      initial_hidden_.data());
      acoustic_embedding.insert(acoustic_embedding.end(),
                                initial_hidden_.begin(), initial_hidden_.end());
      integrate += this_alpha - threshold;
      Scale(p_enc + i * enc_dim, enc_dim, integrate, initial_hidden_.data());
    }
    alpha_cache_[0] = integrate;

    if (acoustic_embedding.empty()) {
      return;  // 本块无 speech token
    }

    // CPU 解码器 (ONNX)
    int32_t num_tokens =
        static_cast<int32_t>(acoustic_embedding.size()) / enc_dim;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> enc_shape{1, num_frames, enc_dim};
    Ort::Value enc_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<float*>(p_enc), num_frames * enc_dim,
        enc_shape.data(), enc_shape.size());

    // decoder.int8.onnx 的长度输入为 int32
    int32_t enc_len_val = num_frames;
    std::array<int64_t, 1> len_shape{1};
    Ort::Value enc_len_tensor = Ort::Value::CreateTensor(
        memory_info, &enc_len_val, 1, len_shape.data(), 1);

    std::array<int64_t, 3> acoustic_shape{1, num_tokens, enc_dim};
    Ort::Value acoustic_tensor = Ort::Value::CreateTensor(
        memory_info, acoustic_embedding.data(), acoustic_embedding.size(),
        acoustic_shape.data(), acoustic_shape.size());

    Ort::Value acoustic_len_tensor = Ort::Value::CreateTensor(
        memory_info, &num_tokens, 1, len_shape.data(), 1);

    if (states_.empty()) {
      int32_t kernel = model_->DecoderKernelSize() - 1;
      std::array<int64_t, 3> state_shape{1, model_->EncoderOutputSize(),
                                         kernel};
      for (int32_t i = 0; i != model_->DecoderNumBlocks(); ++i) {
        Ort::Value s = Ort::Value::CreateTensor<float>(
            model_->Allocator(), state_shape.data(), state_shape.size());
        memset(s.GetTensorMutableData<float>(), 0,
               sizeof(float) * state_shape[0] * state_shape[1] * state_shape[2]);
        states_.push_back(std::move(s));
      }
    }

    auto decoder_out = model_->ForwardDecoder(
        std::move(enc_tensor), std::move(enc_len_tensor),
        std::move(acoustic_tensor), std::move(acoustic_len_tensor),
        std::move(states_));

    states_.clear();
    for (int32_t i = 2; i != static_cast<int32_t>(decoder_out.size()); ++i) {
      states_.push_back(std::move(decoder_out[i]));
    }

    const int64_t* p_ids = decoder_out[1].GetTensorData<int64_t>();
    int64_t n_ids = decoder_out[1].GetTensorTypeAndShapeInfo().GetShape()[1];

    // token→文本组装 (子词拼接 + 中英空格感知):
    // tokens.txt 中以 "@@" 结尾的 token 是子词续接标记 (如 "mon@@" + "day"
    // = "monday", "tom@@" + "or@@" + "row" = "tomorrow" 可链式续接): 去掉
    // 标记后与后续 token 直接相连。空格规则: ASCII 字母/数字后接单词词首
    // (英文词边界 "the day"、中英边界 "tomorrow 是") 补空格; 中文字符后接
    // 拉丁词不补空格 ("是monday"); 中文字符之间不补; 中文标点 (，。！？；：)
    // 前不补。跨 chunk 续接状态由 pending_cont_ 保持。
    std::string text;
    const auto last_char = [&]() -> char {
      if (!text.empty()) {
        return text.back();
      }
      return utterance_.empty() ? '\0' : utterance_.back();
    };
    for (int64_t i = 0; i < n_ids; ++i) {
      if (p_ids[i] == 0) {
        continue;
      }
      auto it = tokens_.find(static_cast<int32_t>(p_ids[i]));
      if (it == tokens_.end()) {
        continue;
      }
      std::string tok = it->second;
      const bool ends_cont =
          tok.size() >= 2 && tok.compare(tok.size() - 2, 2, "@@") == 0;
      if (ends_cont) {
        tok.erase(tok.size() - 2);
      }
      if (!pending_cont_ && !tok.empty()) {
        const char lc = last_char();
        const bool last_ascii_alnum =
            (lc >= 'a' && lc <= 'z') || (lc >= 'A' && lc <= 'Z') ||
            (lc >= '0' && lc <= '9');
        if (last_ascii_alnum) {
          const unsigned char* u =
              reinterpret_cast<const unsigned char*>(tok.data());
          // 中文标点 (，。！？；：) 位于 CJK 全角块与 CJK 符号块
          const bool cjk_punct =
              (u[0] == 0xEF && u[1] == 0xBC &&
               (u[2] == 0x81 || u[2] == 0x8C || u[2] == 0x9A ||
                u[2] == 0x9B || u[2] == 0x9F)) ||
              (u[0] == 0xE3 && u[1] == 0x80 &&
               (u[2] == 0x80 || u[2] == 0x82));
          if (u[0] < 0x80 || !cjk_punct) {
            text += ' ';
          }
        }
      }
      text += tok;
      pending_cont_ = ends_cont;
    }

    if (getenv("SSUB_NPU_DEBUG")) {
      // ORT 编码器对照: 同一特征在 CPU(float/int8) 上的输出, 检验
      // RKNN 转换保真度
      float alpha_sum = 0;
      for (int32_t i = 0; i != num_frames; ++i) {
        alpha_sum += alpha[i];
      }
      auto memory_info2 =
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      std::array<int64_t, 3> x_shape{1, num_frames, feat_dim_};
      Ort::Value x = Ort::Value::CreateTensor(
          memory_info2, const_cast<float*>(frames.data()), frames.size(),
          x_shape.data(), 3);
      int32_t x_len = num_frames;
      std::array<int64_t, 1> l_shape{1};
      Ort::Value x_len_v =
          Ort::Value::CreateTensor(memory_info2, &x_len, 1, l_shape.data(), 1);
      auto ref = model_->ForwardEncoder(std::move(x), std::move(x_len_v));
      const float* ref_enc = ref[0].GetTensorData<float>();
      const float* ref_alpha = ref[2].GetTensorData<float>();
      float enc_diff = 0, enc_diff_sum = 0, ref_enc_abs = 0;
      float rknn_enc_abs = 0, ref_alpha_sum = 0;
      for (int32_t i = 0; i != num_frames * 512; ++i) {
        const float d = std::fabs(ref_enc[i] - enc_copy[i]);
        enc_diff = std::max(enc_diff, d);
        enc_diff_sum += d;
        ref_enc_abs += std::fabs(ref_enc[i]);
        rknn_enc_abs += std::fabs(enc_copy[i]);
      }
      for (int32_t i = 0; i != num_frames; ++i) {
        ref_alpha_sum += ref_alpha[i];
      }
      // 结构签名: 恒等布局余弦相似度 + 逐帧量级剖面。若 RKNN enc 是
      // ORT enc 的某种布局重排, 余弦/剖面会暴露特征; 若值本身错乱, 则均低
      double dot = 0, na = 0, nb = 0;
      for (int32_t i = 0; i != num_frames * 512; ++i) {
        dot += (double)ref_enc[i] * enc_copy[i];
        na += (double)ref_enc[i] * ref_enc[i];
        nb += (double)enc_copy[i] * enc_copy[i];
      }
      float enc_cos = (na > 0 && nb > 0)
                          ? static_cast<float>(dot / std::sqrt(na * nb))
                          : -99.f;
      float ofa[8] = {0}, rfa[8] = {0};
      for (int32_t t = 0; t < num_frames; ++t) {
        float so = 0, sr = 0;
        for (int32_t c = 0; c < 512; ++c) {
          so += std::fabs(ref_enc[t * 512 + c]);
          sr += std::fabs(enc_copy[t * 512 + c]);
        }
        if (t < 6) {
          ofa[t] = so;
          rfa[t] = sr;
        } else if (t == num_frames - 2) {
          ofa[6] = so;
          rfa[6] = sr;
        } else if (t == num_frames - 1) {
          ofa[7] = so;
          rfa[7] = sr;
        }
      }
      fprintf(stderr,
              "npu ref: enc_cos=%.4f "
              "ofa=[%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f] "
              "rfa=[%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f]\n",
              enc_cos, ofa[0], ofa[1], ofa[2], ofa[3], ofa[4], ofa[5], ofa[6],
              ofa[7], rfa[0], rfa[1], rfa[2], rfa[3], rfa[4], rfa[5], rfa[6],
              rfa[7]);
      if (getenv("SSUB_NPU_DUMP_ENC")) {
        static int enc_idx = 0;
        if (enc_idx < 16) {
          char p[200];
          snprintf(p, sizeof(p),
                   "/data/local/tmp/npu/probe/rknn_enc_%03d.bin", enc_idx);
          FILE* fp = fopen(p, "wb");
          if (fp) {
            fwrite(enc_copy.data(), sizeof(float), enc_copy.size(), fp);
            fclose(fp);
          }
          snprintf(p, sizeof(p),
                   "/data/local/tmp/npu/probe/ort_enc_%03d.bin", enc_idx);
          fp = fopen(p, "wb");
          if (fp) {
            fwrite(ref_enc, sizeof(float), num_frames * 512, fp);
            fclose(fp);
          }
          enc_idx++;
        }
      }
      if (getenv("SSUB_NPU_DEBUG")) {
        fprintf(stderr, "npu ref: ort_alpha vec=[");
        for (int32_t i = 0; i != num_frames; ++i) {
          fprintf(stderr, "%s%.3f", i ? "," : "", ref_alpha[i]);
        }
        fprintf(stderr, "]\n");
      }
      fprintf(stderr,
              "npu ref: ort_alpha_sum=%.3f rknn_alpha_sum=%.3f "
              "enc_max_diff=%.4f enc_mean_diff=%.4f "
              "ort_enc_abs=%.1f rknn_enc_abs=%.1f\n",
              ref_alpha_sum, alpha_sum, enc_diff,
              enc_diff_sum / (num_frames * 512), ref_enc_abs, rknn_enc_abs);
    }

    if (!text.empty()) {
      utterance_ += text;
      AsrResult res;
      res.text = utterance_;
      res.isFinal = false;
      if (callback_) {
        callback_(res);
      }
    }
  }

  void DecodeLoop() {
    while (running_) {
      bool ready = false;
      {
        std::lock_guard<std::mutex> lk(stream_mu_);
        ready = mf_->AvailableFrames() >=
                num_processed_ + kChunkFrames;
      }
      if (ready) {
        try {
          ProcessChunk();
        } catch (const std::exception& e) {
          fprintf(stderr, "npu: ProcessChunk exception: %s\n", e.what());
          lastError_ = e.what();
        } catch (...) {
          fprintf(stderr, "npu: ProcessChunk unknown exception\n");
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }

  std::unique_ptr<sherpa_onnx::OnlineParaformerModel> model_;
  std::unique_ptr<ManualFbank> mf_;
  std::map<int32_t, std::string> tokens_;

  rknn_context ctx_ = 0;
  int32_t feat_dim_ = 560;
  rknn_tensor_format in_fmt_ = RKNN_TENSOR_NHWC;
  uint32_t in_elems_ = 0;
  uint32_t alpha_index_ = 2;  // alphas 输出索引 (最后一个输出)
  bool use_ort_ = false;      // true: ORT 编码器 (CPU 模式)

  // 外置 CIF: cif.onnx 在 CPU 上算 alphas (RKNN 内量化 CIF 保真度差)
  Ort::Env cif_env_{ORT_LOGGING_LEVEL_WARNING, "ssub-cif"};
  std::shared_ptr<Ort::Session> cif_sess_;
  bool use_external_cif_ = false;

  // NPU 拼接路线: 前段 RKNN (前 N 块) + 尾部 ORT (N+1..49), 块 N 残差处
  // fp32 交接 (见 README NPU 章节)
  std::unique_ptr<Ort::Session> tail_sess_;
  bool use_split_ = false;

  int32_t num_processed_ = 0;
  std::vector<float> feat_cache_;
  std::vector<float> initial_hidden_;
  std::vector<float> alpha_cache_;
  std::vector<Ort::Value> states_;

  std::string utterance_;
  bool pending_cont_ = false;
  std::string lastError_ = "ok";
  AsrCallback callback_;

  mutable std::mutex stream_mu_;
  std::thread worker_;
  std::atomic<bool> running_{false};
  bool finalized_ = false;
};

}  // namespace

std::unique_ptr<AsrBackend> CreateNpuBackend(bool use_ort)
{
  return std::make_unique<NpuBackend>(use_ort);
}
}  // namespace subtitle

#else  // RKNN 或 sherpa 不可用

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "asr_backend.h"

namespace subtitle {
namespace {
class NpuStub : public AsrBackend {
 public:
  bool Init(const BackendConfig&, AsrCallback) override
  {
    lastError_ = "RKNN runtime was not linked into this build (see README NPU section)";
    return false;
  }
  void Feed(const int16_t*, size_t) override {}
  void Reset() override {}
  std::string LastError() const override { return lastError_; }
 private:
  std::string lastError_;
};
}
std::unique_ptr<AsrBackend> CreateNpuBackend(bool)
{
  return std::make_unique<NpuStub>();
}
}  // namespace subtitle

#endif
