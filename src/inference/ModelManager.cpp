#include "inference/ModelManager.hpp"
#include <iostream>
#include <thread>
#include <chrono>

namespace bonsai {
namespace inference {

ModelManager& ModelManager::getInstance() {
    static ModelManager instance;
    return instance;
}

ModelManager::~ModelManager() {
    if (m_engine) {
        litert_lm_engine_delete(m_engine);
    }
}

bool ModelManager::init(const std::string& modelPath, bool useGpu, bool useVisionGpu, bool useAudioGpu) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_initialized) {
        return true;
    }

    std::cout << "[ModelManager] Initializing engine with model: " << modelPath << std::endl;

    // Pass "CPU" for backends to ensure they are loaded if present in the model.
    LiteRtLmEngineSettings* settings = litert_lm_engine_settings_create(modelPath.c_str(), 
                                                                       useGpu ? "GPU" : "CPU", 
                                                                       useVisionGpu ? "GPU" : "CPU", 
                                                                       useAudioGpu ? "GPU" : "CPU");
    if (!settings) {
        std::cerr << "[ModelManager] Failed to create engine settings." << std::endl;
        return false;
    }

    // Raise the engine's max-sequence-length from the LiteRT-LM default (4096)
    // up to the Gemma 4 E2B LiteRT-LM build's compiled limit (32K).
    litert_lm_engine_settings_set_max_num_tokens(settings, 32768);

    // CPU perf knobs for Pi 4 / aarch64. The Gemma 4 E2B LiteRT-LM build ships
    // both prefill_128 and prefill_1024 signatures; use the larger chunk to
    // amortize per-kernel overhead. F16 activations match XNNPACK's fast path.
    // Speculative decoding leverages the bundled mtp_drafter signature to cut
    // decode latency without a separate draft model.
    litert_lm_engine_settings_set_prefill_chunk_size(settings, 1024);
    litert_lm_engine_settings_set_activation_data_type(settings, 1);  // 1 = F16
    litert_lm_engine_settings_set_enable_speculative_decoding(settings, true);

    m_engine = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);

    if (!m_engine) {
        std::cerr << "[ModelManager] Failed to create engine." << std::endl;
        return false;
    }

    m_modelPath = modelPath;
    m_initialized = true;
    
    std::cout << "[ModelManager] Engine initialized successfully." << std::endl;
    return true;
}

bool ModelManager::isInitialized() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_initialized;
}

void ModelManager::acquireInferenceLock() {
    std::unique_lock<std::mutex> lock(m_inferenceMutex);
    m_inferenceCv.wait(lock, [this] { return !m_isGenerating; });
    m_isGenerating = true;
}

void ModelManager::releaseInferenceLock() {
    // Safety delay to allow LiteRT-LM internal threads to finish cleanup and avoid heap corruption on immediate reuse.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    {
        std::lock_guard<std::mutex> lock(m_inferenceMutex);
        m_isGenerating = false;
    }
    m_inferenceCv.notify_all();
}

} // namespace inference
} // namespace bonsai
