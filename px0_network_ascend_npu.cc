#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "acl/acl.h"
#include "utils/files.h"
#include "utils/bititer.h"
#include "neural/factory.h"
#include "blocking_queue.hpp"
#include "neural/onnx/converter.h"

#define ASCEND_NPU_CACHE_DIR std::string("/dev/shm/.ascend-npu-cache-px")
#define ASCEND_NPU_CORE_ID { 0, 1, 2, 3 }
#define ASCEND_NPU_CORE_NUM 4
#define ASCEND_NPU_BATCH_SIZE 16

#define ASCEND_NPU_DEBUG_MODE false
#define ASCEND_NPU_DEBUG_LOG false

namespace lczero {
namespace {

class AscendNPUNetwork;
class AscendNPUComputation;

enum class AscendNPUTaskType { COMPUTE, LOAD_MODEL, UNLOAD_MODEL };

typedef struct {
  AscendNPUTaskType type;
  AscendNPUComputation* computation;
  uint taskId;
  size_t batchSize;
  void* inputData;
  std::string modelPath;
} ascendNPUTask;

typedef struct {
  uint taskId;
  std::vector<aclFloat16> policy_head_output_;
  std::vector<aclFloat16> wdl_head_output_;
  std::vector<aclFloat16> value_head_output_;
  std::vector<aclFloat16> mlh_head_output_;
} ascendNPUComputeResult;

class AscendNPUComputation : public NetworkComputation {
 public:
  AscendNPUComputation(AscendNPUNetwork* network)
      : network_(network),
        result_queue_(1024 * ASCEND_NPU_CORE_NUM) {}
  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(input);
  }
  void ComputeBlocking() override;
  int GetBatchSize() const override {
    return raw_input_.size();
  }
  float GetQVal(int sample) const override;
  float GetDVal(int sample) const override;
  float GetMVal(int sample) const override;
  float GetPVal(int sample, int move_id) const override;

  AscendNPUNetwork* network_;
  BlockingQueue<ascendNPUComputeResult> result_queue_;

 private:
  std::vector<InputPlanes> raw_input_;
  std::vector<aclFloat16> policy_head_output_;
  std::vector<aclFloat16> wdl_head_output_;
  std::vector<aclFloat16> value_head_output_;
  std::vector<aclFloat16> mlh_head_output_;
};

class AscendNPUNetwork : public Network {
 public:
  AscendNPUNetwork(const WeightsFile& file, std::string modelPath);
  ~AscendNPUNetwork();

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<AscendNPUComputation>(this);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override {
    return ASCEND_NPU_BATCH_SIZE * ASCEND_NPU_CORE_NUM * 2;
  }

  static uint networkCount;
  const NetworkCapabilities capabilities_;
  const std::string modelPath_;
  const bool wdl_head_;
  const bool value_head_;
  const bool mlh_head_;
  static std::mutex* model_lock_;
};
uint AscendNPUNetwork::networkCount = 0;
std::mutex* AscendNPUNetwork::model_lock_ = new std::mutex;

class AscendNPU {
 public:
  static void Init() {
    static bool initialized = false;
    if (initialized) return;
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) throw Exception("[Ascend NPU] Ascend cl init failed!");
    AscendNPU::Log("[Ascend NPU] Ascend cl init OK.");
    BlockingQueue<int> threadStatus(ASCEND_NPU_CORE_NUM);
    auto index = 0;
    for (auto &deviceId : ASCEND_NPU_CORE_ID) {
      modelSync_[index] = new BlockingQueue<bool>(1);
      std::thread worker(
        [&, deviceId, index] {
          auto ret = aclrtSetDevice(deviceId);
          if (ret != ACL_SUCCESS) {
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " init device failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          aclrtContext context;
          ret = aclrtCreateContext(&context, deviceId);
          if (ret != ACL_SUCCESS) {
            aclrtResetDevice(deviceId);
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " create context failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          void* inputMem;
          ret = aclrtMalloc(&inputMem, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9, ACL_MEM_MALLOC_HUGE_FIRST);
          if (ret != ACL_SUCCESS) {
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " malloc device failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          auto inputBuffer = aclCreateDataBuffer(inputMem, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9);
          auto input = aclmdlCreateDataset();
          ret = aclmdlAddDatasetBuffer(input, inputBuffer);
          if (ret != ACL_SUCCESS) {
            aclmdlDestroyDataset(input);
            aclDestroyDataBuffer(inputBuffer);
            aclrtFree(inputMem);
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " add buffer to input dataset failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          void* policyOutputMemHost;
          ret = aclrtMallocHost(&policyOutputMemHost, 2 * ASCEND_NPU_BATCH_SIZE * 2062);
          if (ret != ACL_SUCCESS) {
            aclmdlDestroyDataset(input);
            aclDestroyDataBuffer(inputBuffer);
            aclrtFree(inputMem);
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " malloc policy host failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          void* wdlOutputMemHost;
          ret = aclrtMallocHost(&wdlOutputMemHost, 2 * ASCEND_NPU_BATCH_SIZE * 3);
          if (ret != ACL_SUCCESS) {
            aclrtFreeHost(policyOutputMemHost);
            aclmdlDestroyDataset(input);
            aclDestroyDataBuffer(inputBuffer);
            aclrtFree(inputMem);
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " malloc wdl host failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          void* mlhOutputMemHost;
          ret = aclrtMallocHost(&mlhOutputMemHost, 2 * ASCEND_NPU_BATCH_SIZE);
          if (ret != ACL_SUCCESS) {
            aclrtFreeHost(wdlOutputMemHost);
            aclrtFreeHost(policyOutputMemHost);
            aclmdlDestroyDataset(input);
            aclDestroyDataBuffer(inputBuffer);
            aclrtFree(inputMem);
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " malloc mlh host failed!");
            threadStatus.push(1);
            AscendNPU::stopQueue_.push(true);
            return;
          }
          AscendNPU::Log("[Ascend NPU] Worker thread " + std::to_string(deviceId) + " created successfully.");
          threadStatus.push(0);
          std::map<std::string, uint32_t> modelMap;
          ascendNPUTask computeTask;
          for (;;) {
            ascendNPUComputeResult result;
            computeTask = AscendNPU::taskQueue_.pop();
            if (!AscendNPU::isOk_) {
              if (computeTask.inputData != nullptr) {
                aclrtFreeHost(computeTask.inputData);
              }
              if (computeTask.computation != nullptr) {
                computeTask.computation->result_queue_.push({});
              }
              break;
            }
            if (computeTask.type != AscendNPUTaskType::COMPUTE) {
              AscendNPU::modelGotMutex_->lock();
              if (++AscendNPU::modelGotCount_ == ASCEND_NPU_CORE_NUM) {
                for (size_t i = 0; i < ASCEND_NPU_CORE_NUM; i++) AscendNPU::modelSync_[i]->push(true);
              }
              AscendNPU::modelGotMutex_->unlock();
              AscendNPU::modelSync_[index]->pop();
              if (computeTask.type == AscendNPUTaskType::LOAD_MODEL) { // Load
                uint32_t modelId;
                ret = aclmdlLoadFromFile(computeTask.modelPath.c_str(), &modelId);
                if (ret != ACL_SUCCESS) {
                  AscendNPU::isOk_ = false;
                  AscendNPU::modelResult_.push(false);
                  AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " load model failed!");
                  break;
                }
                modelMap[computeTask.modelPath] = modelId;
                AscendNPU::modelResult_.push(true);
              } else { // Unload
                uint32_t modelId = modelMap[computeTask.modelPath];
                ret = aclmdlUnload(modelId);
                modelMap.erase(computeTask.modelPath);
                AscendNPU::modelResult_.push(ret == ACL_SUCCESS);
              }
              continue;
            }
            ret = aclrtMemcpy(inputMem, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9, computeTask.inputData, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
              aclrtFreeHost(computeTask.inputData);
              AscendNPU::isOk_ = false;
              computeTask.computation->result_queue_.push({});
              AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " memcpy to device failed!");
              break;
            }
            ret = aclrtFreeHost(computeTask.inputData);
            if (ret != ACL_SUCCESS) {
              AscendNPU::isOk_ = false;
              computeTask.computation->result_queue_.push({});
              AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " free input host failed!");
              break;
            }
            bool failed = false;
            auto output = aclmdlCreateDataset();
            for (int i = 0; i < (computeTask.computation->network_->wdl_head_ + computeTask.computation->network_->value_head_ + computeTask.computation->network_->mlh_head_ + 1); i++) {
              auto buffer = aclCreateDataBuffer(nullptr, 0);
              ret = aclmdlAddDatasetBuffer(output, buffer);
              if (ret != ACL_SUCCESS) {
                AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " add buffer to output dataset failed!");
                failed = true;
                break;
              }
            }
            if (failed) {
              for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); i++) {
                aclDestroyDataBuffer(aclmdlGetDatasetBuffer(output, i));
              }
              aclmdlDestroyDataset(output);
              AscendNPU::isOk_ = false;
              computeTask.computation->result_queue_.push({});
              break;
            }
            // Run model
            ret = aclmdlExecute(modelMap[computeTask.computation->network_->modelPath_], input, output);
            if (ret != ACL_SUCCESS) {
              for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); i++) {
                auto buffer = aclmdlGetDatasetBuffer(output, i);
                if (aclGetDataBufferSizeV2(buffer) != 0) {
                  aclrtFree(aclGetDataBufferAddr(buffer));
                }
                aclDestroyDataBuffer(buffer);
              }
              aclmdlDestroyDataset(output);
              AscendNPU::isOk_ = false;
              computeTask.computation->result_queue_.push({});
              AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " execute model failed!");
              break;
            }
            // Read output data
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); i++) {
              auto buffer = aclmdlGetDatasetBuffer(output, i);
              switch (aclGetDataBufferSizeV2(buffer)) {
                case 2 * ASCEND_NPU_BATCH_SIZE * 2062:
                  ret = aclrtMemcpy(policyOutputMemHost, 2 * ASCEND_NPU_BATCH_SIZE * 2062, aclGetDataBufferAddr(buffer), 2 * computeTask.batchSize * 2062, ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret != ACL_SUCCESS) {
                    AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " memcpy from device failed!");
                    failed = true;
                  }
                  break;

                case 2 * ASCEND_NPU_BATCH_SIZE * 3:
                  ret = aclrtMemcpy(wdlOutputMemHost, 2 * ASCEND_NPU_BATCH_SIZE * 3, aclGetDataBufferAddr(buffer), 2 * computeTask.batchSize * 3, ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret != ACL_SUCCESS) {
                    AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " memcpy from device failed!");
                    failed = true;
                  }
                  break;

                case 2 * ASCEND_NPU_BATCH_SIZE:
                  ret = aclrtMemcpy(mlhOutputMemHost, 2 * ASCEND_NPU_BATCH_SIZE, aclGetDataBufferAddr(buffer), 2 * computeTask.batchSize, ACL_MEMCPY_DEVICE_TO_HOST);
                  if (ret != ACL_SUCCESS) {
                    AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " memcpy from device failed!");
                    failed = true;
                  }
                  break;

                default:
                  AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " model output size mismatch!");
                  failed = true;
                  break;
              }
              if (failed) {
                break;
              }
            }
            if (failed) {
              for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); i++) {
                auto buffer = aclmdlGetDatasetBuffer(output, i);
                if (aclGetDataBufferSizeV2(buffer) != 0) {
                  aclrtFree(aclGetDataBufferAddr(buffer));
                }
                aclDestroyDataBuffer(buffer);
              }
              aclmdlDestroyDataset(output);
              AscendNPU::isOk_ = false;
              computeTask.computation->result_queue_.push({});
              break;
            }
            ret = 0;
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); i++) {
              auto buffer = aclmdlGetDatasetBuffer(output, i);
              if (aclGetDataBufferSizeV2(buffer) != 0) {
                auto addr = aclGetDataBufferAddr(buffer);
                ret |= aclrtFree(addr);
              }
              ret |= aclDestroyDataBuffer(buffer);
            }
            ret |= aclmdlDestroyDataset(output);
            if (ret != ACL_SUCCESS) {
              AscendNPU::isOk_ = false;
              computeTask.computation->result_queue_.push({});
              AscendNPU::Log("[Ascend NPU] Thread " + std::to_string(deviceId) + " destroy output dataset failed!");
              break;
            }
            result.taskId = computeTask.taskId;
            for (size_t i = 0; i < computeTask.batchSize * 2062; i++) {
              result.policy_head_output_.push_back(static_cast<aclFloat16*>(policyOutputMemHost)[i]);
            }
            if (computeTask.computation->network_->wdl_head_) {
              for (size_t i = 0; i < computeTask.batchSize * 3; i++) {
                result.wdl_head_output_.push_back(static_cast<aclFloat16*>(wdlOutputMemHost)[i]);
              }
            }
            for (size_t i = 0; i < computeTask.batchSize; i++) {
              (computeTask.computation->network_->value_head_ ? result.value_head_output_ : result.mlh_head_output_).push_back(static_cast<aclFloat16*>(mlhOutputMemHost)[i]);
            }
            computeTask.computation->result_queue_.push(result);
          }
          ret = 0;
          for (auto &model : modelMap) {
            ret |= aclmdlUnload(model.second);
          }
          ret |= aclrtFreeHost(mlhOutputMemHost);
          ret |= aclrtFreeHost(wdlOutputMemHost);
          ret |= aclrtFreeHost(policyOutputMemHost);
          ret |= aclmdlDestroyDataset(input);
          ret |= aclDestroyDataBuffer(inputBuffer);
          ret |= aclrtFree(inputMem);
          ret |= aclrtDestroyContext(context);
          ret |= aclrtResetDevice(deviceId);
          if (ret != 0) {
            AscendNPU::Log("[Ascend NPU] Warning: Some resources on thread " + std::to_string(deviceId) + " failed to be released!");
          }
          AscendNPU::Log("[Ascend NPU] Worker thread " + std::to_string(deviceId) + " exited.");
          AscendNPU::stopQueue_.push(true);
        }
      );
      worker.detach();
      index++;
    }
    for (size_t i = 0; i < ASCEND_NPU_CORE_NUM; i++) {
      if (threadStatus.pop() != 0) {
        Stop();
        throw Exception("[Ascend NPU] Worker thread init failed!");
      }
    }
    initialized = true;
  }

  static std::string GetCacheFile() {
    static bool firstRun = true;
    if (firstRun) {
      if (access(ASCEND_NPU_CACHE_DIR.c_str(), F_OK) != 0) {
        if (mkdir(ASCEND_NPU_CACHE_DIR.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
          throw Exception("[Ascend NPU] Create cache dir failed!");
        }
      }
      uint i = 0;
      while (access((ASCEND_NPU_CACHE_DIR + "/" + std::to_string(i) + ".onnx").c_str(), F_OK) == 0 ||
             access((ASCEND_NPU_CACHE_DIR + "/" + std::to_string(i) + ".om").c_str(), F_OK) == 0) {
        remove((ASCEND_NPU_CACHE_DIR + "/" + std::to_string(i) + ".onnx").c_str());
        remove((ASCEND_NPU_CACHE_DIR + "/" + std::to_string(i) + ".om").c_str());
        i++;
      }
      firstRun = false;
    }
    static uint cacheFileName = 0;
    return ASCEND_NPU_CACHE_DIR + "/" + std::to_string(cacheFileName++);
  }

  static void Stop() {
    static bool stopped = false;
    if (stopped) return;
    stopped = true;
    isOk_ = false;
    for (size_t i = 0; i < ASCEND_NPU_CORE_NUM; i++) { // Activate threads
      taskQueue_.push({ AscendNPUTaskType::COMPUTE, nullptr, 0, 0, nullptr, "" });
    }
    for (size_t i = 0; i < ASCEND_NPU_CORE_NUM; i++) { // Wait all threads exit
      stopQueue_.pop();
    }
    // Clean
    while (!taskQueue_.empty()) {
      auto task = taskQueue_.pop();
      if (task.inputData != nullptr) {
        auto ret = aclrtFreeHost(task.inputData);
        if (ret != ACL_SUCCESS) {
          Log("[Ascend NPU] Warning: Free input host failed!");
        }
      }
      if (task.computation != nullptr) {
        task.computation->result_queue_.push({});
      }
    }
    auto ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
      Log("[Ascend NPU] Warning: Finalize failed!");
    }
  }

  static void Log(std::string msg) {
    static std::mutex logMutex;
    logMutex.lock();
    CERR << msg;
    logMutex.unlock();
  }

  static std::atomic_bool isOk_;
  static BlockingQueue<ascendNPUTask> taskQueue_;
  static BlockingQueue<bool> stopQueue_;
  static uint modelGotCount_;
  static std::mutex* modelGotMutex_;
  static BlockingQueue<bool>* modelSync_[ASCEND_NPU_CORE_NUM];
  static BlockingQueue<bool> modelResult_;
};
std::atomic_bool AscendNPU::isOk_(true);
BlockingQueue<ascendNPUTask> AscendNPU::taskQueue_(1024 * ASCEND_NPU_CORE_NUM);
BlockingQueue<bool> AscendNPU::stopQueue_(ASCEND_NPU_CORE_NUM);
uint AscendNPU::modelGotCount_ = 0;
std::mutex* AscendNPU::modelGotMutex_ = new std::mutex;
BlockingQueue<bool>* AscendNPU::modelSync_[]{};
BlockingQueue<bool> AscendNPU::modelResult_(ASCEND_NPU_CORE_NUM);

AscendNPUNetwork::AscendNPUNetwork(const WeightsFile& file, std::string modelPath)
    : capabilities_{file.format().network_format().input(),
                    file.format().network_format().output(),
                    file.format().network_format().moves_left()},
      modelPath_(modelPath),
      wdl_head_(file.onnx_model().has_output_wdl()),
      value_head_(file.onnx_model().has_output_value()),
      mlh_head_(file.onnx_model().has_output_mlh()) {
  networkCount++;
  model_lock_->lock();
  AscendNPU::modelGotCount_ = 0;
  for (size_t deviceId = 0; deviceId < ASCEND_NPU_CORE_NUM; deviceId++) {
    AscendNPU::taskQueue_.push({ AscendNPUTaskType::LOAD_MODEL, nullptr, 0, 0, nullptr, modelPath_ });
  }
  for (size_t deviceId = 0; deviceId < ASCEND_NPU_CORE_NUM; deviceId++) {
    AscendNPU::modelResult_.pop();
    if (!AscendNPU::isOk_) {
      model_lock_->unlock();
      AscendNPU::Stop();
      throw Exception("[Ascend NPU] Load model failed!");
    }
  }
  model_lock_->unlock();
#if !ASCEND_NPU_DEBUG_MODE
  if (remove(modelPath_.c_str()) != 0) {
    AscendNPU::Log("[Ascend NPU] Warning: Remove temp om file failed!");
  }
#endif
  AscendNPU::Log("[Ascend NPU] New network created successfully.");
}

AscendNPUNetwork::~AscendNPUNetwork() {
  if (AscendNPU::isOk_) {
    bool failed = false;
    model_lock_->lock();
    AscendNPU::modelGotCount_ = 0;
    for (size_t deviceId = 0; deviceId < ASCEND_NPU_CORE_NUM; deviceId++) {
      AscendNPU::taskQueue_.push({ AscendNPUTaskType::UNLOAD_MODEL, nullptr, 0, 0, nullptr, modelPath_ });
    }
    for (size_t deviceId = 0; deviceId < ASCEND_NPU_CORE_NUM; deviceId++) {
      if (!AscendNPU::modelResult_.pop()) {
        failed = true;
      }
    }
    model_lock_->unlock();
    if (failed) {
      AscendNPU::Log("[Ascend NPU] Warning: Some model failed to be released!");
    }
    if (--networkCount == 0) {
      AscendNPU::Stop();
    }
  }
}

void AscendNPUComputation::ComputeBlocking() {
  size_t batchOffset = 0;
  size_t totalBatchSize = raw_input_.size();
  size_t batchBatchSize;
  uint taskId = 0;
  while (batchOffset != totalBatchSize) {
    batchBatchSize = std::min(totalBatchSize - batchOffset, static_cast<size_t>(ASCEND_NPU_BATCH_SIZE));
    // Prepare input data
    auto ret = aclrtSetDevice(std::vector<int>(ASCEND_NPU_CORE_ID)[0]);
    if (ret != ACL_SUCCESS) {
      AscendNPU::Stop();
      throw Exception("[Ascend NPU] Computation set device failed!");
    }
    void* inputData;
    ret = aclrtMallocHost(&inputData, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9);
    if (ret != ACL_SUCCESS) {
      aclrtResetDevice(std::vector<int>(ASCEND_NPU_CORE_ID)[0]);
      AscendNPU::Stop();
      throw Exception("[Ascend NPU] Computation malloc input host failed!");
    }
    // Never forget to initialize the memory!!!
    ret = aclrtMemset(inputData, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9, 0, 2 * ASCEND_NPU_BATCH_SIZE * kInputPlanes * 10 * 9);
    if (ret != ACL_SUCCESS) {
      aclrtFreeHost(inputData);
      aclrtResetDevice(std::vector<int>(ASCEND_NPU_CORE_ID)[0]);
      AscendNPU::Stop();
      throw Exception("[Ascend NPU] Computation memset input host failed!");
    }
    ret = aclrtResetDevice(std::vector<int>(ASCEND_NPU_CORE_ID)[0]);
    if (ret != ACL_SUCCESS) {
      aclrtFreeHost(inputData);
      AscendNPU::Stop();
      throw Exception("[Ascend NPU] Computation reset device failed!");
    }
    aclFloat16* iter = static_cast<aclFloat16*>(inputData);
    for (size_t i = batchOffset; i < batchOffset + batchBatchSize; i++) { // 16
      for (const auto& plane : raw_input_[i]) { // 124
        aclFloat16 value = aclFloatToFloat16(plane.value);
        for (auto bit : IterateBits(plane.mask)) { // 10 * 9
          *(iter + bit) = value;
        }
        iter += 90;
      }
    }
#if ASCEND_NPU_DEBUG_LOG
    if (batchOffset == 0) {
      std::stringstream s;
      s << "ASCEND NPU: " << aclFloat16ToFloat(aclFloatToFloat16(2049.0)) << std::endl;
      s << iter << " " << iter + 1 << std::endl;
      for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 9; j++) {
          s << aclFloat16ToFloat(static_cast<aclFloat16*>(inputData)[(i * 10) + j]) << " ";
        }
        s << std::endl;
      }
      AscendNPU::Log(s.str());
    }
#endif
    AscendNPU::taskQueue_.push({ AscendNPUTaskType::COMPUTE, this, taskId, batchBatchSize, inputData, "" });
    batchOffset += batchBatchSize;
    taskId++;
  }
  std::map<uint, ascendNPUComputeResult> resultMap;
  ascendNPUComputeResult result;
  for (size_t i = 0; i < taskId; i++) {
    result = result_queue_.pop();
    if (!AscendNPU::isOk_) {
      AscendNPU::Stop();
      throw Exception("[Ascend NPU] Computation task failed!");
    }
    resultMap[result.taskId] = result;
  }
  for (size_t i = 0; i < taskId; i++) {
    result = resultMap[i];
    policy_head_output_.insert(policy_head_output_.end(), result.policy_head_output_.begin(), result.policy_head_output_.end());
    wdl_head_output_.insert(wdl_head_output_.end(), result.wdl_head_output_.begin(), result.wdl_head_output_.end());
    value_head_output_.insert(value_head_output_.end(), result.value_head_output_.begin(), result.value_head_output_.end());
    mlh_head_output_.insert(mlh_head_output_.end(), result.mlh_head_output_.begin(), result.mlh_head_output_.end());
  }
#if ASCEND_NPU_DEBUG_LOG
  std::stringstream s;
  s << "ASCEND NPU OUTPUT:" << std::endl;
  for (size_t i = 0; i < 10; i++) {
    for (size_t j = 0; j < 9; j++) {
      s << aclFloat16ToFloat(policy_head_output_.data()[(i * 10) + j]) << " ";
    }
    s << std::endl;
  }
  AscendNPU::Log(s.str());
#endif
}

float AscendNPUComputation::GetQVal(int sample) const {
  if (network_->wdl_head_) {
    return aclFloat16ToFloat(wdl_head_output_[sample * 3 + 0]) - aclFloat16ToFloat(wdl_head_output_[sample * 3 + 2]);
  } else {
    return aclFloat16ToFloat(value_head_output_[sample]);
  }
}

float AscendNPUComputation::GetDVal(int sample) const {
  if (!network_->wdl_head_) return 0.0f;
  return aclFloat16ToFloat(wdl_head_output_[sample * 3 + 1]);
}

float AscendNPUComputation::GetMVal(int sample) const {
  if (!network_->mlh_head_) return 0.0f;
  return aclFloat16ToFloat(mlh_head_output_[sample]);
}

float AscendNPUComputation::GetPVal(int sample, int move_id) const {
  return aclFloat16ToFloat(policy_head_output_[sample * 2062 + move_id]);
}

}  // namespace

std::unique_ptr<Network> MakeAscendNPUNetwork(
    const std::optional<WeightsFile>& weights, [[maybe_unused]] const OptionsDict& options) {
  if (!weights) throw Exception("[Ascend NPU] Requires a network file!");
  AscendNPU::Init();
  WeightsFile onnxModel = *weights;
  if (!onnxModel.has_onnx_model()) {
    WeightsToOnnxConverterOptions converterOptions;
    converterOptions.batch_size = ASCEND_NPU_BATCH_SIZE;
    converterOptions.data_type = WeightsToOnnxConverterOptions::DataType::kFloat16;
    onnxModel = ConvertWeightsToOnnx(onnxModel, converterOptions);
  }
#if !ASCEND_NPU_DEBUG_MODE
  const auto& onnx = onnxModel.onnx_model();
  if (!onnx.has_input_planes()) {
    throw Exception("[Ascend NPU] Model doesn't have input planes defined!");
  }
  if (!onnx.has_output_policy()) {
    throw Exception("[Ascend NPU] Model doesn't have policy head defined!");
  }
  if (!onnx.has_output_wdl() && !onnx.has_output_value()) {
    throw Exception("[Ascend NPU] Model doesn't have value head!");
  }
  if (onnx.has_output_wdl() && onnx.has_output_value()) {
    throw Exception("[Ascend NPU] Model has two value heads!");
  }
  if (onnx.has_output_value() && onnx.has_output_mlh()) {
    throw Exception("[Ascend NPU] Model has value head and mlh head, which will cause the model to be unable to distinguish them by the output size!");
  }
  auto cacheFile = AscendNPU::GetCacheFile();
  WriteStringToFile(cacheFile + ".onnx", onnx.model());
  AscendNPU::Log("[Ascend NPU] Start convert model '" + cacheFile + ".onnx' to '" + cacheFile + ".om'...");
  if (system(("atc --model \"" + cacheFile + ".onnx\" --framework 5 --input_shape \"/input/planes:" + std::to_string(ASCEND_NPU_BATCH_SIZE) + "," + std::to_string(kInputPlanes) + ",10,9\" --input_format=NCHW --output \"" + cacheFile + "\" --soc_version Ascend310").c_str()) != 0) {
    remove((cacheFile + ".onnx").c_str());
    throw Exception("[Ascend NPU] Convert failed!");
  }
  AscendNPU::Log("[Ascend NPU] Convert success.");
  if (remove((cacheFile + ".onnx").c_str()) != 0) {
    AscendNPU::Log("[Ascend NPU] Warning: Remove temp onnx file failed!");
  }
  return std::make_unique<AscendNPUNetwork>(onnxModel, cacheFile + ".om");
#else
  return std::make_unique<AscendNPUNetwork>(onnxModel, "./.ascend-npu-cache-px/0.om");
#endif
}

REGISTER_NETWORK("ascend_npu", MakeAscendNPUNetwork, -99999)

}  // namespace lczero
