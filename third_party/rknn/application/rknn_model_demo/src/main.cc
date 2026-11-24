// RKNN runtime verification demo for OpenHarmony.
//
// Verifies that the bundled librknnrt.so can load and run a converted
// .rknn model on the device:
//   1. load the .rknn model from argv[1]
//   2. rknn_init + query input/output tensor attributes
//   3. feed dummy input (zeroes), rknn_run, fetch and print output stats
//   4. release everything cleanly
//
// Usage: rknn_model_demo <model.rknn>
#include <cstdio>
#include <cstring>
#include <vector>

#include <rknn_api.h>

namespace {

std::vector<char> ReadFile(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
        return {};
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(size > 0 ? static_cast<size_t>(size) : 0);
    if (!buf.empty()) {
        size_t rd = fread(buf.data(), 1, buf.size(), f);
        if (rd != buf.size()) {
            buf.clear();
        }
    }
    fclose(f);
    return buf;
}

void PrintTensorAttr(const rknn_tensor_attr& attr, const char* kind)
{
    printf("  %s[%u]: n=%d c=%d h=%d w=%d, fmt=%d, type=%d, size=%u\n",
           kind, attr.index, attr.dims[0], attr.dims[1], attr.dims[2],
           attr.dims[3], attr.fmt, attr.type, attr.size);
}

}  // namespace

int main(int argc, char** argv)
{
    // 未指定模型时使用 hardware_npu2 示例的 RK3588 mobilenet 模型
    const char* modelPath = (argc >= 2) ? argv[1] : "model/RK3588/mobilenet_v1.rknn";

    std::vector<char> model = ReadFile(modelPath);
    if (model.empty()) {
        printf("ERROR: cannot read model file: %s\n", modelPath);
        return 1;
    }
    printf("model loaded: %s (%zu bytes)\n", modelPath, model.size());

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model.data(), model.size(), 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_init failed, ret=%d\n", ret);
        return 1;
    }
    printf("rknn_init OK\n");

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("model inputs: %u\n", ret == RKNN_SUCC ? io_num.n_input : 0);

    // 输入属性查询与构造
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret != RKNN_SUCC) {
        printf("ERROR: query input attr failed, ret=%d\n", ret);
        return 1;
    }
    PrintTensorAttr(in_attr, "input");

    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (ret != RKNN_SUCC) {
        printf("ERROR: query output attr failed, ret=%d\n", ret);
        return 1;
    }
    PrintTensorAttr(out_attr, "output");

    // 构造输入并推理: 按模型输入属性计算 float32 缓冲大小
    uint32_t elems = 1;
    for (int i = 0; i < RKNN_MAX_DIMS; i++) {
        if (in_attr.dims[i] > 0) {
            elems *= in_attr.dims[i];
        }
    }
    std::vector<float> dummy(elems, 0.0f);

    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_FLOAT32;
    in.fmt = in_attr.fmt;
    in.size = elems * sizeof(float);
    in.buf = dummy.data();
    in.pass_through = 0;
    ret = rknn_inputs_set(ctx, 1, &in);
    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_inputs_set failed, ret=%d\n", ret);
        return 1;
    }

    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_run failed, ret=%d\n", ret);
        return 1;
    }
    printf("rknn_run OK (NPU inference)\n");

    rknn_output out;
    memset(&out, 0, sizeof(out));
    out.index = 0;
    out.want_float = 1;
    out.is_prealloc = 0;
    ret = rknn_outputs_get(ctx, 1, &out, nullptr);
    if (ret != RKNN_SUCC) {
        printf("ERROR: rknn_outputs_get failed, ret=%d\n", ret);
        return 1;
    }

    const float* values = reinterpret_cast<const float*>(out.buf);
    uint32_t count = out.size / sizeof(float);
    printf("output tensors: %u floats, first values:", count);
    for (uint32_t i = 0; i < count && i < 5; i++) {
        printf(" %.6f", values[i]);
    }
    printf("\n");
    rknn_outputs_release(ctx, 1, &out);

    rknn_destroy(ctx);
    printf("rknn_model_demo PASSED\n");
    return 0;
}
