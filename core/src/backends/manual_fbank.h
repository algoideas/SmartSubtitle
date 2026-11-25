#pragma once
// 手工 fbank 特征提取: 与 tools/host_ref.py (已在主机验证可正确解码
// 中英混合文本) 的数学完全一致, 用于替代 knf OnlineStream 特征链。
// 参数: 16kHz, 25ms 帧长/10ms 帧移, 512 点 FFT, 80 梅滤波,
// low_freq=20, high_freq=8000(奈奎斯特), povey 窗, 预加重 0.97,
// 首样本不预加重, ln(floor 1e-15)。
#include <cmath>
#include <cstdint>
#include <vector>

namespace subtitle {

class ManualFbank {
 public:
  ManualFbank() {
    int32_t low = 20, high = SR_ / 2;
    float ml = 1127.0f * std::log(1.0f + low / 700.0f);
    float mh = 1127.0f * std::log(1.0f + high / 700.0f);
    std::vector<float> pts(NMEL_ + 2);
    for (int32_t i = 0; i < NMEL_ + 2; ++i) {
      pts[i] = ml + (mh - ml) * i / float(NMEL_ + 1);
    }
    std::vector<int32_t> bins(NMEL_ + 2);
    for (int32_t i = 0; i < NMEL_ + 2; ++i) {
      float f = 700.0f * (std::exp(pts[i] / 1127.0f) - 1.0f);
      bins[i] = static_cast<int32_t>(std::floor((NFFT_ + 1) * f / SR_));
    }
    mel_.assign(NMEL_ * (NFFT_ / 2 + 1), 0.0f);
    for (int32_t b = 0; b < NMEL_; ++b) {
      int32_t l = bins[b], c = bins[b + 1], r = bins[b + 2];
      if (c == l) c = l + 1;
      if (r == c) r = c + 1;
      for (int32_t k = l; k < c; ++k) {
        if (k < static_cast<int32_t>(NFFT_ / 2 + 1)) {
          mel_[b * (NFFT_ / 2 + 1) + k] = (k - l) / float(c - l);
        }
      }
      for (int32_t k = c; k < r; ++k) {
        if (k < static_cast<int32_t>(NFFT_ / 2 + 1)) {
          mel_[b * (NFFT_ / 2 + 1) + k] = (r - k) / float(r - c);
        }
      }
    }
    win_.resize(FL_);
    for (int32_t n = 0; n < FL_; ++n) {
      win_[n] = std::pow(
          0.5f - 0.5f * std::cos(2.0f * M_PI * n / (FL_ - 1)), 0.85f);
    }
  }

  void Feed(const int16_t* pcm, size_t samples) {
    pcm_.insert(pcm_.end(), pcm, pcm + samples);
  }

  size_t SampleCount() const { return pcm_.size(); }

  // 当前可产出的完整 fbank 帧数
  int32_t AvailableFrames() const {
    return static_cast<int32_t>(pcm_.size()) >= FL_
               ? static_cast<int32_t>((pcm_.size() - FL_) / FS_) + 1
               : 0;
  }

  // 返回从第 start 帧起 n 帧 (每帧 80 维) 的 fbank; 样本不足返回空
  std::vector<float> GetFrames(int32_t start, int32_t n) {
    int32_t need = (start + n - 1) * FS_ + FL_;
    if (static_cast<int32_t>(pcm_.size()) < need) {
      return {};
    }
    std::vector<float> out;
    out.reserve(n * NMEL_);
    for (int32_t i = start; i < start + n; ++i) {
      // 分帧 + 去直流。保持原始 int16 量级: 与已验证可正确解码的
      // 主机参考管线(tools/host_ref.py)一致, 配合 CMVN(v-neg_mean)
      // 归一化尺度会使特征偏离训练分布导致无 token 输出
      float frame[FL_];
      const float* base = pcm_.data() + i * FS_;
      float sum = 0.0f;
      for (int32_t k = 0; k < FL_; ++k) sum += base[k];
      float dc = sum / FL_;
      for (int32_t k = 0; k < FL_; ++k) frame[k] = base[k] - dc;
      // 预加重 (首样本保持)
      for (int32_t k = FL_ - 1; k > 0; --k) {
        frame[k] -= 0.97f * frame[k - 1];
      }
      // 加窗
      for (int32_t k = 0; k < FL_; ++k) frame[k] *= win_[k];
      // FFT 功率谱
      std::vector<float> pow_spec = PowerSpectrum(frame);
      // 梅滤波 + ln
      for (int32_t b = 0; b < NMEL_; ++b) {
        float e = 0.0f;
        const float* w = mel_.data() + b * (NFFT_ / 2 + 1);
        for (int32_t k = 0; k < static_cast<int32_t>(NFFT_ / 2 + 1); ++k) {
          e += w[k] * pow_spec[k];
        }
        out.push_back(std::log(std::max(e, 1e-15f)));
      }
    }
    return out;
  }

 private:
  // 512 点基-2 FFT (实输入, 虚部置 0), 返回 0..256 bin 的功率谱
  std::vector<float> PowerSpectrum(const float* frame) {
    std::vector<float> re(NFFT_, 0.0f), im(NFFT_, 0.0f);
    for (int32_t i = 0; i < FL_; ++i) re[i] = frame[i];
    // 位反转
    for (int32_t i = 1, j = 0; i < NFFT_; ++i) {
      int32_t bit = NFFT_ >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        std::swap(re[i], re[j]);
        std::swap(im[i], im[j]);
      }
    }
    for (int32_t len = 2; len <= NFFT_; len <<= 1) {
      float ang = -2.0f * M_PI / len;
      float wr = std::cos(ang), wi = std::sin(ang);
      for (int32_t i = 0; i < NFFT_; i += len) {
        float cr = 1.0f, ci = 0.0f;
        for (int32_t k = 0; k < len / 2; ++k) {
          float ur = re[i + k], ui = im[i + k];
          float vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          float vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ur + vr;
          im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr;
          im[i + k + len / 2] = ui - vi;
          float ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = ncr;
        }
      }
    }
    std::vector<float> pow_spec(NFFT_ / 2 + 1);
    for (int32_t k = 0; k <= NFFT_ / 2; ++k) {
      pow_spec[k] = re[k] * re[k] + im[k] * im[k];
    }
    return pow_spec;
  }

  static constexpr int32_t SR_ = 16000;
  static constexpr int32_t FL_ = 400;   // 25ms
  static constexpr int32_t FS_ = 160;   // 10ms
  static constexpr int32_t NFFT_ = 512;
  static constexpr int32_t NMEL_ = 80;

  std::vector<float> pcm_;
  std::vector<float> mel_;
  std::vector<float> win_;
};

}  // namespace subtitle
