#include "engine/framework/assets/tensor_source.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Explicit, CPU-only comparison of official checkpoint tensors with an existing
// lossless safetensors preparation. No model sessions or GPU backend are created.
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        const std::filesystem::path original(argv[1]), prepared(argv[2]);
        struct Pair { const char * raw; const char * safe; const char * raw_prefix; const char * safe_prefix; };
        const std::vector<Pair> files = {
            {"gpt.pth", "gpt.safetensors", "", ""},
            {"s2mel.pth", "s2mel.safetensors", "net", ""},
            {"feat1.pt", "feat1.safetensors", "", ""},
            {"feat2.pt", "feat2.safetensors", "", ""},
            {"wav2vec2bert_stats.pt", "wav2vec2bert_stats.safetensors", "", ""},
            {"auxiliary/campplus/campplus_cn_common.bin", "campplus.safetensors", "", "speaker_encoder"},
            {"auxiliary/bigvgan/bigvgan_generator.pt", "bigvgan/model.safetensors", "generator", ""},
        };
        for (const auto & pair : files) {
            const auto raw = engine::assets::open_tensor_source(original / pair.raw, pair.raw_prefix);
            const auto safe = engine::assets::open_tensor_source(prepared / pair.safe, pair.safe_prefix);
            const auto tensors = safe->tensors();
            for (const auto & tensor : tensors) {
                const auto a = raw->require_tensor_data(tensor.name);
                const auto b = safe->require_tensor_data(tensor.name);
                if (a.metadata.dtype != b.metadata.dtype || a.metadata.shape != b.metadata.shape || a.bytes != b.bytes) {
                    throw std::runtime_error(std::string(pair.raw) + ": unequal tensor " + tensor.name);
                }
            }
            std::cout << pair.raw << ": " << tensors.size() << " tensors byte-identical\n";
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
