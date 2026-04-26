#ifndef CPUINFER_OPERATOR_PAGEDMOE_HPP
#define CPUINFER_OPERATOR_PAGEDMOE_HPP

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common.hpp"
#include "pagedmoe_c_api.h"

class PAGEDMOE_MOE : public MoE_Interface {
 private:
  struct RuntimeHolder {
    explicit RuntimeHolder(PagedMoeRuntime* runtime) : runtime(runtime) {}
    ~RuntimeHolder() {
      if (runtime != nullptr) {
        log_stats();
        pagedmoe_runtime_destroy(runtime);
        runtime = nullptr;
      }
    }

    RuntimeHolder(const RuntimeHolder&) = delete;
    RuntimeHolder& operator=(const RuntimeHolder&) = delete;

    void log_stats_if_needed(uint64_t added_token_calls) {
      std::lock_guard<std::mutex> lock(stats_log_mutex);
      token_calls_since_last_log += added_token_calls;
      if (token_calls_since_last_log < kStatsLogIntervalTokenCalls) {
        return;
      }
      token_calls_since_last_log = 0;
      log_stats_unlocked();
    }

    void log_stats() {
      std::lock_guard<std::mutex> lock(stats_log_mutex);
      log_stats_unlocked();
    }

    PagedMoeRuntime* runtime = nullptr;

   private:
    static constexpr uint64_t kStatsLogIntervalTokenCalls = 1000;

    void log_stats_unlocked() {
      if (runtime == nullptr) {
        return;
      }
      PagedMoeRuntimeStats stats{};
      if (pagedmoe_runtime_get_stats(runtime, &stats) != PAGEDMOE_OK || stats.token_calls == 0) {
        return;
      }
      printf(
          "PagedMoe runtime stats: token_calls=%llu route_slots=%llu route_unique_requests=%llu "
          "total_possible_bitplane_reads=%llu submitted_reads=%llu completed_reads=%llu "
          "submitted_compute_tasks=%llu completed_compute_tasks=%llu read_job_cache_hit_rate=%.4f\n",
          static_cast<unsigned long long>(stats.token_calls), static_cast<unsigned long long>(stats.route_slots),
          static_cast<unsigned long long>(stats.route_unique_requests),
          static_cast<unsigned long long>(stats.total_possible_bitplane_reads),
          static_cast<unsigned long long>(stats.submitted_reads),
          static_cast<unsigned long long>(stats.completed_reads),
          static_cast<unsigned long long>(stats.submitted_compute_tasks),
          static_cast<unsigned long long>(stats.completed_compute_tasks), stats.read_job_cache_hit_rate);
      fflush(stdout);
    }

    std::mutex stats_log_mutex;
    uint64_t token_calls_since_last_log = 0;
  };

  static std::mutex& runtime_cache_mutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<std::string, std::weak_ptr<RuntimeHolder>>& runtime_cache() {
    static std::unordered_map<std::string, std::weak_ptr<RuntimeHolder>> cache;
    return cache;
  }

  static std::string runtime_cache_key(const GeneralMOEConfig& config) {
    return config.path + "|cache=" + std::to_string(config.pagedmoe_cache_size_bytes) +
           "|codebook=" + std::to_string(config.pagedmoe_codebook_workers) +
           "|bitplane=" + std::to_string(config.pagedmoe_bitplane_workers) +
           "|compute=" + std::to_string(config.pagedmoe_compute_threads) +
           "|pin=" + std::to_string(config.pagedmoe_pin_compute_workers) +
           "|layers=" + std::to_string(config.pagedmoe_num_layers) +
           "|experts=" + std::to_string(config.expert_num) +
           "|hidden=" + std::to_string(config.hidden_size) +
           "|intermediate=" + std::to_string(config.intermediate_size) +
           "|blocks=" + std::to_string(config.pagedmoe_num_blocks);
  }

 public:
  explicit PAGEDMOE_MOE(const GeneralMOEConfig& config) : config(config) {
    printf("PagedMoe MoE layer %d, storage: %s\n", config.layer_idx, config.path.c_str());
  }

  ~PAGEDMOE_MOE() = default;

  PAGEDMOE_MOE(const PAGEDMOE_MOE&) = delete;
  PAGEDMOE_MOE& operator=(const PAGEDMOE_MOE&) = delete;

  void load_weights() {
    if (runtime_holder_ != nullptr && runtime_holder_->runtime != nullptr) {
      loaded_ = true;
      return;
    }
    if (config.path.empty()) {
      throw std::runtime_error("PagedMoe_MOE requires MOEConfig.path to point to pagedmoe storage_root");
    }

    const std::string cache_key = runtime_cache_key(config);
    std::lock_guard<std::mutex> lock(runtime_cache_mutex());
    auto& cache = runtime_cache();
    auto found = cache.find(cache_key);
    if (found != cache.end()) {
      runtime_holder_ = found->second.lock();
      if (runtime_holder_ != nullptr && runtime_holder_->runtime != nullptr) {
        loaded_ = true;
        return;
      }
      cache.erase(found);
    }

    PagedMoeRuntime* runtime = nullptr;
    PagedMoeRuntimeConfig runtime_config{};
    runtime_config.storage_root = config.path.c_str();
    runtime_config.cache_size_bytes = config.pagedmoe_cache_size_bytes;
    runtime_config.codebook_workers = config.pagedmoe_codebook_workers;
    runtime_config.bitplane_workers = config.pagedmoe_bitplane_workers;
    runtime_config.compute_threads = config.pagedmoe_compute_threads;
    runtime_config.pin_compute_workers = config.pagedmoe_pin_compute_workers;
    runtime_config.num_layers = config.pagedmoe_num_layers;
    runtime_config.num_experts = config.expert_num;
    runtime_config.hidden_size = config.hidden_size;
    runtime_config.intermediate_size = config.intermediate_size;
    runtime_config.num_blocks = config.pagedmoe_num_blocks;

    auto status = pagedmoe_runtime_create(&runtime_config, &runtime);
    if (status != PAGEDMOE_OK) {
      throw std::runtime_error(std::string("pagedmoe_runtime_create failed: ") + pagedmoe_last_error());
    }
    runtime_holder_ = std::make_shared<RuntimeHolder>(runtime);
    cache[cache_key] = runtime_holder_;
    loaded_ = true;
  }

  void warm_up() {
    if (!loaded_) {
      load_weights();
    }
    std::vector<uint16_t> input(config.hidden_size, 0);
    std::vector<uint16_t> output(config.hidden_size, 0);
    std::vector<int64_t> expert_ids(config.num_experts_per_tok);
    std::vector<float> weights(config.num_experts_per_tok, 1.0f / config.num_experts_per_tok);
    for (int i = 0; i < config.num_experts_per_tok; ++i) {
      expert_ids[i] = i % config.expert_num;
    }
    forward(1, config.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  PagedMoeRuntimeStats stats() const {
    PagedMoeRuntimeStats stats{};
    if (runtime_holder_ == nullptr || runtime_holder_->runtime == nullptr) {
      return stats;
    }
    auto status = pagedmoe_runtime_get_stats(runtime_holder_->runtime, &stats);
    if (status != PAGEDMOE_OK) {
      throw std::runtime_error(std::string("pagedmoe_runtime_get_stats failed: ") + pagedmoe_last_error());
    }
    return stats;
  }

  void reset_stats() {
    if (runtime_holder_ == nullptr || runtime_holder_->runtime == nullptr) {
      return;
    }
    auto status = pagedmoe_runtime_reset_stats(runtime_holder_->runtime);
    if (status != PAGEDMOE_OK) {
      throw std::runtime_error(std::string("pagedmoe_runtime_reset_stats failed: ") + pagedmoe_last_error());
    }
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental = false) override {
    if (!loaded_ || runtime_holder_ == nullptr || runtime_holder_->runtime == nullptr) {
      throw std::runtime_error("PagedMoe_MOE runtime is not loaded");
    }
    if (incremental) {
      throw std::runtime_error("PagedMoe_MOE does not support deferred/incremental MoE merge");
    }
    if (qlen < 0 || k <= 0) {
      throw std::runtime_error("PagedMoe_MOE received invalid qlen or top-k");
    }

    const int64_t* route_experts = expert_ids;
    std::vector<int64_t> filtered_experts;
    if (config.gpu_experts_mask != nullptr) {
      const size_t route_len = static_cast<size_t>(qlen) * static_cast<size_t>(k);
      filtered_experts.assign(expert_ids, expert_ids + route_len);
      for (size_t i = 0; i < route_len; ++i) {
        if (config.should_skip_expert(filtered_experts[i])) {
          filtered_experts[i] = -1;
        }
      }
      route_experts = filtered_experts.data();
    }

    auto status = pagedmoe_runtime_execute_batch_sum_bf16(
        runtime_holder_->runtime, static_cast<size_t>(config.layer_idx), static_cast<size_t>(qlen), static_cast<size_t>(k),
        route_experts, weights, reinterpret_cast<const uint16_t*>(input), reinterpret_cast<uint16_t*>(output));
    if (status != PAGEDMOE_OK) {
      throw std::runtime_error(std::string("pagedmoe_runtime_execute_batch_sum_bf16 failed: ") +
                               pagedmoe_last_error());
    }
    runtime_holder_->log_stats_if_needed(static_cast<uint64_t>(qlen));
  }

  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input,
                       intptr_t output, bool incremental = false) {
    forward(*reinterpret_cast<int*>(qlen_ptr), k, reinterpret_cast<const int64_t*>(expert_ids),
            reinterpret_cast<const float*>(weights), reinterpret_cast<const void*>(input), reinterpret_cast<void*>(output),
            incremental);
  }

  GeneralMOEConfig config;

 private:
  std::shared_ptr<RuntimeHolder> runtime_holder_;
  bool loaded_ = false;
};

#endif
