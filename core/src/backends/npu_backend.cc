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

#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-paraformer-model.h"
#include "sherpa-onnx/csrc/online-stream.h"

#include "asr_backend.h"

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

    // 2) 特征提取配置: paraformer 要求样本保持原始 int16 量级
    sherpa_onnx::FeatureExtractorConfig feat_cfg;
    feat_cfg.feature_dim = 80;
    feat_cfg.normalize_samples = false;
    stream_ = std::make_unique<sherpa_onnx::OnlineStream>(feat_cfg);

    // 3) RKNN 编码器
    if (LoadModel(config.modelDir + "/paraformer-encoder.rknn") != RKNN_SUCC) {
      lastError_ = "failed to load paraformer-encoder.rknn from " +
                   config.modelDir;
      return false;
    }
    rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0);

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
      fprintf(stderr, "npu: encoder outputs=%u alpha_index=%u\n",
              io_num.n_output, alpha_index_);
    }

    feat_dim_ = static_cast<int32_t>(model_->NegativeMean().size());

    running_ = true;
    worker_ = std::thread(&NpuBackend::DecodeLoop, this);
    return true;
  }

  void Feed(const int16_t* pcm, size_t samples) override {
    std::vector<float> floats(samples);
    // normalize_samples=false: 保持原始 int16 量级
    for (size_t i = 0; i < samples; i++) {
      floats[i] = pcm[i];
    }
    std::lock_guard<std::mutex> lk(stream_mu_);
    stream_->AcceptWaveform(kSampleRate, floats.data(),
                            static_cast<int32_t>(floats.size()));
  }

  void Reset() override {
    std::lock_guard<std::mutex> lk(stream_mu_);
    stream_->Reset();
    num_processed_ = 0;
    feat_cache_.clear();
    initial_hidden_.clear();
    alpha_cache_.assign(1, 0.0f);
    states_.clear();
    utterance_.clear();
  }

  std::string LastError() const override {
    std::lock_guard<std::mutex> lk(stream_mu_);
    return lastError_;
  }

  ~NpuBackend() {
    running_ = false;
    if (worker_.joinable()) {
      worker_.join();
    }
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
        p[k] = (p[k] + neg_mean[k]) * inv_stddev[k];
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
    {
      std::lock_guard<std::mutex> lk(stream_mu_);
      frames = stream_->GetFrames(num_processed_, kChunkFrames);
      num_processed_ += kChunkFrames - 1;
    }
    if (getenv("SSUB_NPU_DEBUG")) {
      fprintf(stderr, "npu: ProcessChunk frames=%zu\n", frames.size() / 80);
    }

    frames = ApplyLFR(frames);
    ApplyCMVN(&frames);
    PositionalEncoding(&frames, num_processed_ / model_->LfrWindowShift());
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

    // 左右文缓存: 首块补零, 之后与上一块尾部重叠
    if (feat_cache_.empty()) {
      feat_cache_.resize((kLeftChunkFrames + kRightChunkFrames) * feat_dim_, 0);
    }
    frames.insert(frames.begin(), feat_cache_.begin(), feat_cache_.end());
    std::copy(frames.end() - feat_cache_.size(), frames.end(),
              feat_cache_.begin());

    int32_t num_frames = static_cast<int32_t>(frames.size()) / feat_dim_;

    // RKNN 编码器推理
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_FLOAT32;
    in.fmt = in_fmt_;
    in.size = frames.size() * sizeof(float);
    in.buf = frames.data();
    in.pass_through = 0;
    if (rknn_inputs_set(ctx_, 1, &in) != RKNN_SUCC) {
      lastError_ = "rknn_inputs_set failed";
      return;
    }
    if (getenv("SSUB_NPU_DUMP_CALIB")) {
      // 导出真实特征作为 int8 量化校准数据 (最多 32 块)
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
    if (rknn_run(ctx_, nullptr) != RKNN_SUCC) {
      lastError_ = "rknn_run failed";
      fprintf(stderr, "npu: rknn_run failed\n");
      return;
    }

    // 只请求 enc(0) 与 alphas(最后), 跳过 int64 的 enc_len
    rknn_output outs[2];
    memset(outs, 0, sizeof(outs));
    outs[0].index = 0;
    outs[0].want_float = 1;
    outs[0].is_prealloc = 0;
    outs[1].index = alpha_index_;
    outs[1].want_float = 1;
    outs[1].is_prealloc = 0;
    if (rknn_outputs_get(ctx_, 2, outs, nullptr) != RKNN_SUCC) {
      lastError_ = "rknn_outputs_get failed";
      fprintf(stderr, "npu: rknn_outputs_get failed\n");
      return;
    }

    const float* p_enc =
        reinterpret_cast<const float*>(outs[0].buf);  // [1, num_frames, 512]
    int32_t enc_dim = 512;
    const float* p_alpha =
        reinterpret_cast<const float*>(outs[1].buf);  // [1, num_frames]

    // 拷贝 enc 数据后立即释放 RKNN 输出缓冲, 避免悬垂引用
    std::vector<float> enc_copy(p_enc, p_enc + num_frames * enc_dim);
    rknn_outputs_release(ctx_, 2, outs);
    p_enc = enc_copy.data();

    // CIF: 左右文边界上的 alpha 置零 (与 sherpa 一致)
    std::vector<float> alpha(p_alpha, p_alpha + num_frames);
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

    std::string text;
    for (int64_t i = 0; i < n_ids; ++i) {
      if (p_ids[i] == 0) {
        continue;
      }
      auto it = tokens_.find(static_cast<int32_t>(p_ids[i]));
      if (it != tokens_.end()) {
        text += it->second;
      }
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
      float enc_diff = 0, ref_enc_abs = 0, ref_alpha_sum = 0;
      for (int32_t i = 0; i != num_frames * 512; ++i) {
        enc_diff = std::max(enc_diff, std::fabs(ref_enc[i] - enc_copy[i]));
        ref_enc_abs += std::fabs(ref_enc[i]);
      }
      for (int32_t i = 0; i != num_frames; ++i) {
        ref_alpha_sum += ref_alpha[i];
      }
      fprintf(stderr,
              "npu ref: ort_alpha_sum=%.3f rknn_alpha_sum=%.3f "
              "enc_max_diff=%.4f ort_enc_abs_sum=%.1f\n",
              ref_alpha_sum, alpha_sum, enc_diff, ref_enc_abs);
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
        ready = stream_->NumFramesReady() >=
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
  std::unique_ptr<sherpa_onnx::OnlineStream> stream_;
  std::map<int32_t, std::string> tokens_;

  rknn_context ctx_ = 0;
  int32_t feat_dim_ = 560;
  rknn_tensor_format in_fmt_ = RKNN_TENSOR_NHWC;
  uint32_t in_elems_ = 0;
  uint32_t alpha_index_ = 2;  // alphas 输出索引 (最后一个输出)

  int32_t num_processed_ = 0;
  std::vector<float> feat_cache_;
  std::vector<float> initial_hidden_;
  std::vector<float> alpha_cache_;
  std::vector<Ort::Value> states_;

  std::string utterance_;
  std::string lastError_ = "ok";
  AsrCallback callback_;

  mutable std::mutex stream_mu_;
  std::thread worker_;
  std::atomic<bool> running_{false};
};

}  // namespace

std::unique_ptr<AsrBackend> CreateNpuBackend()
{
  return std::make_unique<NpuBackend>();
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
std::unique_ptr<AsrBackend> CreateNpuBackend()
{
  return std::make_unique<NpuStub>();
}
}  // namespace subtitle

#endif
