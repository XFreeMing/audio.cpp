#pragma once

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace minitts::app {

struct AppRequest {
    std::string id;
    engine::runtime::TaskRequest request;
};

struct AppRequestResult {
    std::string id;
    engine::runtime::TaskResult result;
    double wall_ms = 0.0;
};

// An additive observation. A completed callback is not necessarily a WAV;
// wav_written is unknown when the callback has not returned successfully.
struct AppRequestTiming {
    std::string id;
    size_t index = 0;
    double prepare_ms = 0.0;
    std::optional<double> run_ms;
    std::optional<double> callback_ms;
    std::optional<double> full_ms;
    bool prepare_completed = false;
    bool run_completed = false;
    bool callback_attempted = false;
    bool callback_completed = false;
    std::optional<bool> wav_written;
};

struct AudioChapter {
    std::string id;
    int64_t start_sample = 0;
    int64_t end_sample = 0;
};

struct AppBatchRequest {
    std::vector<AppRequest> requests;
};

struct AppBatchResult {
    std::vector<AppRequestResult> results;
    std::optional<engine::runtime::AudioBuffer> merged_audio;
    std::vector<AudioChapter> chapters;
    double prepare_ms = 0.0;
    double session_wall_ms = 0.0;
};

enum class AudioMergeMode {
    None,
    Concat,
};

AudioMergeMode parse_audio_merge_mode(const std::string & value);

AppBatchResult run_offline_batch(
    engine::runtime::IVoiceTaskSession & session,
    engine::runtime::IOfflineVoiceTaskSession & offline,
    const AppBatchRequest & batch,
    AudioMergeMode audio_merge_mode,
    const std::function<bool(size_t, const AppRequestResult &)> & on_result = {},
    const std::function<void(const AppRequestTiming &)> & on_timing = {});

}  // namespace minitts::app
