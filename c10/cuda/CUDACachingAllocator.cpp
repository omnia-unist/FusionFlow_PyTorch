#include <string.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <unistd.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/UniqueVoidPtr.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include "ATen/native/ReduceAllOps.h"
#include "c10/core/Device.h"
#include <sys/msg.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>

#define  SHML_SIZE  1000
#define  KEY_REFERENCE 1024
struct shared_block {
  cudaIpcMemHandle_t my_handle;
  void* ptr;
  size_t size;
  int shml_index;
  bool sharable;
};
struct shmid_ds shminfo;
using shared_block = struct shared_block;

template<size_t SIZEOFSHMSEG>
class SharedMemory {
 public:
  explicit SharedMemory(key_t shmkey)
  : shm_address(nullptr), shmkey(shmkey) {
    if (shmkey == 0)
      return;
    shmid = shmget(shmkey, SIZEOFSHMSEG, 0666);
    if (shmid == -1) {
      shmid = shmget(shmkey, SIZEOFSHMSEG, 0666 | IPC_CREAT | IPC_EXCL);
      shm_address = shmat(shmid, NULL, 0);
      memset(shm_address, 0, SIZEOFSHMSEG);
    }
    if (shmid == -1) {
      log("Shmid error");
      perror("Shmid Error: ");
      assert(0);
    }
    if (shm_address == nullptr) {
      shm_address = shmat(shmid, NULL, 0);
      memset(shm_address, 0, SIZEOFSHMSEG);
    }
    if (shm_address == NULL) {
      log("Shmat error");
      perror("Shmat Error: ");
      assert(0);
    }
    log("SharedMemory Constructor success!");
  }

  ~SharedMemory() {
    if (shm_address)
      shmdt(shm_address);
    // Mansur: Current Version is to manually
    // delete Semaphores/SharedMemory using "ipcrm"
    // rc = shmctl(shmid, IPC_RMID, NULL);
  }

  void* ptr() {
    log("Ptr");
    return shm_address;
  }

  void read(void* dst, size_t bytes) {
    if (shm_address == nullptr)
      return;
    log("Read");
    memcpy(dst, shm_address, bytes);
  }

  void write(void* dst, size_t bytes) {
    if (shm_address == nullptr)
      return;
    log("Write");
    memcpy(shm_address, dst, bytes);
  }

  void log(const char* s) {
    fprintf(stdout, "[PyTorch, pid %d, shmkey %d] %s\n", getpid(), shmkey, s);
    fflush(stdout);
  }

 private:
  int shmid;
  void* shm_address;
  key_t shmkey;
};



namespace c10 {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

namespace cuda {
namespace CUDACachingAllocator {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
//   To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//


namespace {

using stream_set = std::unordered_set<cuda::CUDAStream>;

constexpr size_t kMinBlockSize  =      512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize     =  1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer   =  2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer   = 20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge    =  2097152; // round up large allocations to 2 MiB

typedef std::bitset<static_cast<size_t>(StatType::NUM_TYPES)> StatTypes;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT(stat.current >= 0, "Negative tracked stat in CUDA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

void update_stat_array(StatArray& stat_array, int64_t amount, const StatTypes& stat_types) {
  for (size_t stat_type = 0; stat_type < stat_types.size(); ++stat_type) {
    if (stat_types[stat_type]) {
      update_stat(stat_array[stat_type], amount);
    }
  }
}

struct Block;
typedef bool (*Comparison)(const Block*, const Block*);
typedef std::set<Block*, Comparison> BlockPool;

struct Block {
  int           device;       // gpu
  cudaStream_t  stream;       // allocation stream
  stream_set    stream_uses;  // streams on which the block was used
  size_t        size;         // block size in bytes
  BlockPool*    pool;         // owning memory pool
  void*         ptr;          // memory address
  bool          allocated;    // in-use flag
  Block*        prev;         // prev block if split from a larger allocation
  Block*        next;         // next block if split from a larger allocation
  int           event_count;  // number of outstanding CUDA events
  bool          dali_block;   // indicator for dali free list memory
  int           shared_index; // index
  cudaIpcMemHandle_t block_cuda_handler;
  bool lending_candidate = false;

  Block(int device, cudaStream_t stream, size_t size, BlockPool* pool, void* ptr) :
    device(device), stream(stream), stream_uses(), size(size), pool(pool),
    ptr(ptr), allocated(0), prev(nullptr), next(nullptr), event_count(0), dali_block(false), shared_index(-1) { 
      // cudaIpcGetMemHandle(&block_cuda_handler, ptr);
    }

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size) :
    device(device), stream(stream), stream_uses(), size(size), pool(nullptr),
    ptr(nullptr), allocated(0), prev(nullptr), next(nullptr), event_count(0), dali_block(false), shared_index(-1) { 
      // cudaIpcGetMemHandle(&block_cuda_handler, ptr);
    }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

static bool BlockComparator(const Block* a, const Block* b)
{
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

struct AllocParams {
  AllocParams(int device, size_t size, cudaStream_t stream, BlockPool* pool, size_t alloc_size,
              DeviceStats& stats) :
    search_key(device, stream, size),
    pool(pool),
    alloc_size(alloc_size),
    block(nullptr),
    err(cudaSuccess) {}

  int device() { return search_key.device; }
  cudaStream_t stream() { return search_key.stream; }
  size_t size() { return search_key.size; }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types;
  cudaError_t err;
};

} // namespace

class DeviceCachingAllocator {

 private:

  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;
  
  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;
  
  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream
  std::unordered_set<Block*> active_blocks;

  // outstanding cuda events
  std::deque<std::pair<cudaEvent_t, Block*>> cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;
  
  int retval = 0;

  SharedMemory<SHML_SIZE * sizeof(shared_block)> shared_memory_;

  int iteration = 0;

  int device_id;

  shared_block shared_list[SHML_SIZE];
  
  int shml_idx=0;

  // int num_shared_blocks = 0;
  std::vector<std::pair<char*, char*>> borrowed_region;
  std::vector<Block*> marked_blocks;

  size_t minimum_lending_block_size;

 public:


  DeviceCachingAllocator() :
      large_blocks(BlockComparator),
      small_blocks(BlockComparator),
      shared_memory_(0) {
  }

  DeviceCachingAllocator(int i) :      
      large_blocks(BlockComparator),
      small_blocks(BlockComparator),
      shared_memory_((i + 1) * KEY_REFERENCE) {
    // C10_CUDA_CHECK(cudaGetDevice(&(this->device_id)));
    this->device_id = i;
    memset(shared_list, 0, sizeof(shared_list));
    printf("%p\n", (this->shared_list));
    printf("init device allocator on %d\n", this->device_id);

    char* minimum_lending_block_size_str = std::getenv("minimum_lending_block_size");
    if (!minimum_lending_block_size_str) {
      minimum_lending_block_size = 50 * (1 << 21);
    }
    else {
      minimum_lending_block_size = std::atoll(minimum_lending_block_size_str);
    }
  }
  
  ~DeviceCachingAllocator() {
  }

  void specific_index_write(cudaIpcMemHandle_t& handler, size_t size, int* index, void* ptr, bool allocated) {
    size_t offset;
    int result = 1;
    int writable = 0;
    void* shmaddress;
    if (*index == -1 && this->shml_idx < SHML_SIZE) {
      cudaIpcGetMemHandle(&handler, ptr);
      *index = this->shml_idx++;
    }
    if (*index != -1) {
      memcpy(&(this->shared_list[*index].my_handle),&handler, sizeof(cudaIpcMemHandle_t));
      this->shared_list[*index].size = size;
      this->shared_list[*index].ptr = ptr;
      // fflush(stdout);
      if (allocated) {
        this->shared_list[*index].sharable = false;
      }
      else {
        this->shared_list[*index].sharable = true;
      }
      this->shared_list[*index].shml_index = rand();  // *index;
      offset = *index * sizeof(shared_block);
      if (this->shared_list[*index].sharable) {
        shmaddress = shared_memory_.ptr();
        memcpy((char *)shmaddress + offset, (char*)&(this->shared_list[*index]), sizeof(shared_block));
      }
    } 
  }

  bool is_lending_candidate(const Block* block) {
    for (auto borrowed_segment: borrowed_region) {
      if (max(borrowed_segment.first, (char*)block->ptr) < min(borrowed_segment.second, (char*)block->ptr + block->size)) {
        printf("Lending Candidate YES\n");
        fflush(stdout);
        return true;
      }
    }
    return false;
  }

  void mark_shared_memory() {
    int cnt = 0;
    marked_blocks.clear();
    for (const auto& block : large_blocks) {
      if (is_lending_candidate(block)) {
        if (!block->allocated) {
          block->allocated = true;
          block->lending_candidate = true;
          marked_blocks.push_back(block);
          cnt++;
        }
        else {
          printf("Large Be or not to be, thats the question\n");
          fflush(stdout);
        }
      }
    }
    for (const auto& block : small_blocks) {
      if (is_lending_candidate(block)) {
        if (!block->allocated) {
          block->allocated = true;
          block->lending_candidate = true;
          marked_blocks.push_back(block);
          cnt++;
        }
        else {
          printf("Large Be or not to be, thats the question\n");
          fflush(stdout);
        }
      }
    }
    for (const auto& block: marked_blocks) {
      block->pool->erase(block);
    }
    // if (cnt != num_shared_blocks) {
    //   printf("SOME LARGE BLOCK GOT SPLIT! %d %d", cnt, num_shared_blocks);
    //   fflush(stdout);
    // }
  }

  void unmark_shared_memory() {
    int cnt = 0;
    for (const auto& block : large_blocks) {
      if (is_lending_candidate(block) && block->lending_candidate) {
        block->allocated = false;
        block->lending_candidate = false;
        cnt++;
      }
    }
    for (const auto& block : small_blocks) {
      if (is_lending_candidate(block) && block->lending_candidate) {
        block->allocated = false;
        block->lending_candidate = false;
        cnt++;
      }
    }
    for (const auto& block: marked_blocks) {
      block->pool->insert(block);
    }
    marked_blocks.clear();
    // if (cnt != num_shared_blocks) {
    //   printf("SOME LARGE BLOCK GOT SPLIT! %d %d", cnt, num_shared_blocks);
    //   fflush(stdout);
    // }
  }

  void renew_shared_memory() {
    if (large_blocks.empty())
      return;
    void* shmaddress;
    shmaddress = shared_memory_.ptr();
    if (shmaddress == nullptr)
      return;
    memset(shmaddress, 0, sizeof(this->shared_list));
    cudaEvent_t event;
    cudaEventCreate(&event);
    int index_limit = 0;
    size_t total_mem = 0;
    for (const auto& block : large_blocks) {
      if (block->allocated)
        continue;
      total_mem += block->size;
      if (index_limit == SHML_SIZE) {
        break;
      }
      block->shared_index = -1;
      // printf("LargeBlocks contain %p, with size %lu. The index is %d. The allocated info is %d\n", block->ptr, block->size, block->shared_index, block->allocated);
      // if (block->size > 4800000) {50 * (1 << 21)
      if (block->size > minimum_lending_block_size) {  // 100 MB at least
        //block->lending_candidate = true;
        // num_shared_blocks++;
        
        // Block must be 2MB Aligned according to IPC Safety Guide!
        size_t alignment = (1 << 21);
        size_t aligned_size = block->size;
        uintptr_t b_ptr = (uintptr_t)block->ptr;
        size_t lshift = (alignment - b_ptr % alignment);

        b_ptr += lshift;
        aligned_size -= lshift;
        aligned_size -= aligned_size % alignment;

        borrowed_region.push_back(std::make_pair((char*)b_ptr, (char*)b_ptr + aligned_size));
        specific_index_write(block->block_cuda_handler, aligned_size, &(block->shared_index), (void*)b_ptr, block->allocated);
        cudaEventRecord(event, block->stream);
        bool is_ready = false;
        while (!is_ready) {
          cudaError_t e = cudaEventQuery(event);
          switch (e) {
              case cudaSuccess:
                is_ready = true;
                break;
              case cudaErrorNotReady:
                is_ready = false;
                break;
              case cudaErrorCudartUnloading:
                cudaGetLastError();
                is_ready = true;
                break;
              default:
                throw e;
          }
        }
        std::stringstream ss;
        ss << "[specific_index_write]PyTorch index: " << block->shared_index;
        ss << " PyTorch size: " << aligned_size;
        ss << " PyTorch address: " << std::hex << (void*)b_ptr;
        ss << " Total Mem: " << std::dec << total_mem;
        shared_memory_.log(ss.str().c_str());
        index_limit++;
      }
    }
    /*
    for (const auto& block : small_blocks) {
      if (block->allocated)
        continue;
      total_mem += block->size;
      if (index_limit == SHML_SIZE) {
        break;
      }
      block->shared_index = -1;
      // printf("LargeBlocks contain %p, with size %lu. The index is %d. The allocated info is %d\n", block->ptr, block->size, block->shared_index, block->allocated);
      // if (block->size > 4800000) {
      if (block->size > 0) {
        specific_index_write(block->block_cuda_handler, block->size, &(block->shared_index), block->ptr, block->allocated);
        cudaEventRecord(event, block->stream);
        bool is_ready = false;
        while (!is_ready) {
          cudaError_t e = cudaEventQuery(event);
          switch (e) {
              case cudaSuccess:
                is_ready = true;
                break;
              case cudaErrorNotReady:
                is_ready = false;
                break;
              case cudaErrorCudartUnloading:
                cudaGetLastError();
                is_ready = true;
                break;
              default:
                throw e;
          }
        }
        std::stringstream ss;
        ss << "[specific_index_write]PyTorch index: " << block->shared_index;
        ss << " PyTorch size: " << block->size;
        ss << " PyTorch address: " << std::hex << block->ptr;
        ss << " Total Mem: " << std::dec << total_mem;
        shared_memory_.log(ss.str().c_str());
        index_limit++;
      }
    }
    */
    cudaEventDestroy(event);
    std::ofstream outFile("/home/tykim/manticore/reimplement-ds-analyzer/iteration.txt");
    if (outFile.is_open()) {
      outFile << iteration;
      outFile.close();
    }

    iteration++;
    // std::stringstream ss;
    // ss << "[iteration" << iteration;
    // ss << "]********************************************";
    // shared_memory_.log(ss.str().c_str());
    printf("[iteration %d, pid %d]--------------------------------\n", iteration, getpid());
    fflush(stdout);
  //   for (const auto& block : small_blocks) {
  //     printf("SmallBlocks contain %p, with size %lu. The index is %d. The allocated info is %d\n", block->ptr, block->size, block->shared_index, block->allocated);
  //   }
  //   fflush(stdout);
    this->shml_idx = 0;
    
    // printf("************RENEW is called\n");
    // printf("The shared memory ID is %d on GPU %d\n", this->shmid, this->device_id);
    // fflush(stdout);
  }
  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(int device, size_t size, cudaStream_t stream)
  {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    // process outstanding cudaEvents
    process_events();
    size = round_size(size);
    auto& pool = get_pool(size);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    
    bool block_found =
      // Search pool
      get_free_block(params)
      // Trigger callbacks and retry search
      || (trigger_free_memory_callbacks(params) && get_free_block(params))
      // Attempt allocate
      || alloc_block(params, false)
      // Free all non-split cached blocks and retry alloc.
      || (free_cached_blocks() && alloc_block(params, true));

      if(!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

      size_t device_free;
      size_t device_total;
      C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      stats.num_ooms += 1;

      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(CUDAOutOfMemoryError, false,
        "CUDA out of memory. Tried to allocate ", format_size(alloc_size),
        " (GPU ", device, "; ",
        format_size(device_total), " total capacity; ",
        format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
        " already allocated; ",
        format_size(device_free), " free; ",
        allowed_info,
        format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
        " reserved in total by PyTorch)");
    }

    TORCH_INTERNAL_ASSERT(params.err == cudaSuccess &&
                          params.block != nullptr &&
                          params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (should_split(block, size)) {
      remaining = block;

      block = new Block(device, stream, size, &pool, block->ptr);
      
      // block->lending_candidate = remaining->lending_candidate;
      // block->lending_candidate = remaining->dali_block;
      // if (block->lending_candidate)
      //   num_shared_blocks++;

      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      pool.insert(remaining);

      if (already_split) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit block,
        // size remaining->size bytes.
        update_stat_array(stats.inactive_split_bytes, remaining->size, params.stat_types);
        update_stat_array(stats.inactive_split, 1, params.stat_types);
      }
    } else if (already_split) {
      // An already-split block is becoming active
      update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
      update_stat_array(stats.inactive_split, -1, params.stat_types);
    }

    block->allocated = true;
    active_blocks.insert(block);

    c10::reportMemoryUsageToProfiler(
        block, block->size, c10::Device(c10::DeviceType::CUDA, device));

    update_stat_array(stats.allocation, 1, params.stat_types);
    update_stat_array(stats.allocated_bytes, block->size, params.stat_types);
    update_stat_array(stats.active, 1, params.stat_types);
    update_stat_array(stats.active_bytes, block->size, params.stat_types);
    
    return block;
  }

  void free(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    
    block->allocated = false;

    c10::reportMemoryUsageToProfiler(
        block, -block->size, c10::Device(c10::DeviceType::CUDA, block->device));

    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    update_stat_array(stats.allocation, -1, {stat_types});
    update_stat_array(stats.allocated_bytes, -block->size, {stat_types});

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block);
    }
  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
      block = block->prev;
    }
    void *basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(Block* block, cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    // printf("device_free %lu and device_total %lu\n", device_free, device_total);
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    free_cached_blocks();
  }

  /** Retrieves info (total size + largest block) of the memory cache **/
  void cacheInfo(size_t* total, size_t* largest)
  {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest == 0) {  // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes;
      cudaMemGetInfo(largest,  // Use free memory as an optimistic initial guess of *largest
                     &tmp_bytes);
    }
    cache_info_aux(large_blocks, total, largest);
    cache_info_aux(small_blocks, total, largest);
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (size_t statType = 0; statType < static_cast<size_t>(StatType::NUM_TYPES); ++statType) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
    }
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially VERY expensive. **/
  std::vector<SegmentInfo> snapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.is_large = (head_block->pool == &large_blocks);

      const Block* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0);

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
        }

        block = block->next;
      }
    }

    std::sort(result.begin(), result.end(), [](const SegmentInfo& a, const SegmentInfo& b) {
      return a.address < b.address;
    });

    return result;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    }
  }

 private:

  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(blocks.end(), small_blocks.begin(), small_blocks.end());
    blocks.insert(blocks.end(), large_blocks.begin(), large_blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block)
  {
    TORCH_INTERNAL_ASSERT(!block->allocated && block->event_count == 0);
    size_t original_block_size = block->size;
    bool dali_mem = block->dali_block;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    if(!dali_mem) {
      const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
      for (Block* merge_candidate : merge_candidates) {
        const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
        if (subsumed_size > 0) {
          net_change_inactive_split_blocks -= 1;
          net_change_inactive_split_size -= subsumed_size;
        }
      }
    }

    active_blocks.erase(block);
    if(!dali_mem){ 
      pool.insert(block);

      if (block->is_split()) {
        net_change_inactive_split_blocks += 1;
        net_change_inactive_split_size += block->size;
      }
    }

    StatTypes stat_types;
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, stat_types);
    update_stat_array(stats.active, -1, stat_types);
    update_stat_array(stats.active_bytes, -original_block_size, stat_types);
  }

  /** combine previously split blocks. returns the size of the subsumed block, or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool)
  {
    if (!src || src->allocated || src->event_count > 0) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    // if (src->lending_candidate && dst->lending_candidate)
    //   num_shared_blocks--;
    // dst->lending_candidate |= src->lending_candidate;
    pool.erase(src);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size) {
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatType get_stat_type_for_pool(const BlockPool& pool) {
    if (&pool == &small_blocks) {
      return StatType::SMALL_POOL;
    } else if (&pool == &large_blocks) {
      return StatType::LARGE_POOL;
    } else {
      AT_ERROR("get_stat_type_for_pool: invalid pool");
    }
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool == &small_blocks) {
      return remaining >= kMinBlockSize;
    } else if (block->pool == &large_blocks) {
      return remaining > kSmallSize;
    } else {
      AT_ERROR("should_split: invalid pool");
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.lower_bound(&p.search_key);
    if (it == pool.end() || (*it)->stream != p.stream())
      return false;
    p.block = *it;
    pool.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
        FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  bool alloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction && total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      p.err = cudaMalloc(&ptr, size);
      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear CUDA's
          //   internal error state.
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH will take
          //   over to throw a helpful exception. The user can choose to catch the exception,
          //   free some stuff in their script, and attempt their allocation again.
          //   In this case, we can also forgive and clear CUDA's internal error state.
          cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    int temp_value = 0;
    update_stat_array(stats.segment, 1, p.stat_types);
    update_stat_array(stats.reserved_bytes, size, p.stat_types);

    // p.block came from new, not cudaMalloc.  It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
  }

  bool free_cached_blocks()
  {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks
    free_blocks(large_blocks);
    free_blocks(small_blocks);
    memset(this->shared_list, 0, sizeof(this->shared_list));
    this->shml_idx = 0;
    void* shmaddress = shared_memory_.ptr();
    memcpy((char*)shmaddress, (char*)(this->shared_list), sizeof(this->shared_list));
    printf("EmptyCache called -> %lu , index %d\n", sizeof(this->shared_list), this->shml_idx);
    return true;
  }

  void free_blocks(BlockPool& blocks)
  {
    // Frees all non-split blocks
    auto it = blocks.begin();

    while (it != blocks.end()) {
      Block* block = *it;
      if (!block->prev && !block->next) {
        // printf("CUDA Free on block with index -> %d\n", block->shared_index);
        if (is_lending_candidate(block)) {
          printf("Do not cudaFree lending candidate!\n");
          fflush(stdout);
        } else {
          C10_CUDA_CHECK(cudaFree((void*)block->ptr));
        }
        total_allocated_memory -= block->size;

        StatTypes stat_types;
        stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
        stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
        update_stat_array(stats.segment, -1, stat_types);
        update_stat_array(stats.reserved_bytes, -block->size, stat_types);

        auto cur = it;
        ++it;
        blocks.erase(cur);
        delete block;
      } else {
        ++it;
      }
    }
  }
  cudaEvent_t create_event_internal() {
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  void free_event_internal(cudaEvent_t event) {
    C10_CUDA_CHECK(cudaEventDestroy(event));
  }

  void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.

    for (auto& e : cuda_events) {
      cudaEvent_t event = e.first;
      Block* block = e.second;

      C10_CUDA_CHECK(cudaEventSynchronize(event));
      free_event_internal(event);

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block)
  {
    int prev_device;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto it = streams.begin(); it != streams.end(); ++it) {
      C10_CUDA_CHECK(cudaSetDevice(it->device_index()));

      cudaEvent_t event = create_event_internal();
      C10_CUDA_CHECK(cudaEventRecord(event, it->stream()));

      block->event_count++;
      cuda_events.emplace_back(event, block);
    }

    C10_CUDA_CHECK(cudaSetDevice(prev_device));
  }

  void process_events()
  {
    // Process outstanding cudaEvents. Events that are completed are removed
    // from the queue, and the 'event_count' for the corresponding allocation
    // is decremented. Stops at the first event which has not been completed.
    // Since events on different devices or streams may occur out of order,
    // the processing of some events may be delayed.
    while (!cuda_events.empty()) {
      auto& e = cuda_events.front();
      cudaEvent_t event = e.first;
      Block* block = e.second;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) {
        // ignore and clear the error if not ready
        cudaGetLastError();
        break;
      } else if (err != cudaSuccess) {
        C10_CUDA_CHECK(err);
      }

      free_event_internal(event);

      block->event_count--;
      if (block->event_count == 0) {
        free_block(block);
      }
      cuda_events.pop_front();
    }
  }

  // Accumulates sizes of all memory blocks for given device in given pool
  void cache_info_aux(BlockPool& blocks, size_t* total, size_t* largest)
  {
    for (const auto& block : blocks) {
      size_t blocksize = block->size;
      *total += blocksize;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }
};

class THCCachingAllocator {

 private:

  std::mutex mutex;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // lock around calls to cudaFree (to prevent deadlocks with NCCL)
  mutable std::mutex cuda_free_mutex;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
// destructor
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  std::mutex* getCudaFreeMutex() const {
    return &cuda_free_mutex;
  }
  Block* get_allocated_block(void *ptr, bool remove=false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void init(int device_count) {
    int size = device_allocator.size();
    int resize = 0;
    int retval = 0;
    printf("size is %d, device count is %d\n", size, device_count);
    if (size < device_count) {
      device_allocator.resize(device_count);
      resize = device_allocator.size();
      printf("resize is %d, device count is %d\n", resize, device_count);
      for (int i = size; i < device_count; i++) {
        device_allocator[i] = std::unique_ptr<DeviceCachingAllocator>(new DeviceCachingAllocator(i));
      }
    }    
    // printf("init device allocator on %d\n", device_id);
  }
  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    // printf("malloc at %d with size %lu\n",device,size);
    TORCH_INTERNAL_ASSERT(
        0 <= device && device < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, int device) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && device < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction  && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    int activated_device;
    cudaGetDevice (&activated_device);
    if (activated_device != device) {
        cudaSetDevice(device);
    }
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void emptyCache() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->emptyCache();
    // printf("EmptyCache called\n");
  }

  void renew_shared_memory() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->renew_shared_memory();
  }

  void mark_shared_memory() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->mark_shared_memory();
  }

  void unmark_shared_memory() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++)
      device_allocator[i]->unmark_shared_memory();
  }

  void* getBaseAllocation(void* ptr, size_t* outSize)
  {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      AT_ERROR("invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    device_allocator[block->device]->recordStream(block, stream);
  }

  std::vector<SegmentInfo> snapshot() {
    std::vector<SegmentInfo> result;
    int count = device_allocator.size();
    for (int i = 0; i < count; i++) {
      auto snap = device_allocator[i]->snapshot();
      result.insert(result.end(), snap.begin(), snap.end());
    }

    return result;
  }
};

THCCachingAllocator caching_allocator;

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cudaMalloc.  This setting is useful when debugging GPU memory
// errors, since the caching allocator foils cuda-memcheck.
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  C10_CUDA_CHECK(cudaFree(ptr));
}

// NB: I decided not to fold this into THCCachingAllocator, because the latter
// has a lot more methods and it wasn't altogether clear that they should
// actually be publicly exposed
struct CudaCachingAllocator : public Allocator {  
  DataPtr allocate(size_t size) const override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (forceUncachedAllocator()) {
      C10_CUDA_CHECK(cudaMalloc(&r, size));
      return {r, r, &uncached_delete, Device(DeviceType::CUDA, device)};
    }
    if (size != 0) {
      caching_allocator.malloc(&r, device, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &raw_delete, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    return &raw_delete;
  }
};

CudaCachingAllocator device_allocator;

Allocator* get(void)
{
  return &device_allocator;
}

void init(int device_count) {
  caching_allocator.init(device_count);
}

void setMemoryFraction(double fraction, int device) {
  caching_allocator.setMemoryFraction(fraction, device);
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  caching_allocator.device_allocator[dev_id]->cacheInfo(cachedAndFree, largestBlock);
}

void* getBaseAllocation(void *ptr, size_t *size)
{
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(const DataPtr& ptr, cuda::CUDAStream stream)
{
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex()
{
  return caching_allocator.getCudaFreeMutex();
}

static inline void assertValidDevice(int device) {
  int device_num = caching_allocator.device_allocator.size();
  TORCH_CHECK(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats getDeviceStats(int device) {
  assertValidDevice(device);
  return caching_allocator.device_allocator[device]->getStats();
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
  caching_allocator.device_allocator[device]->resetAccumulatedStats();
}

void resetPeakStats(int device) {
  assertValidDevice(device);
  caching_allocator.device_allocator[device]->resetPeakStats();
}

std::vector<SegmentInfo> snapshot() {
  return caching_allocator.snapshot();
}

//
// In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
// is called by the receiving process to map the CUDA memory from the sending
// process into its own address space.
//
// CUDA IPC only allows sharing a big memory block associated with a cudaIpcMemHandle_t
// and it can be opened only **once** per context per process. There can be
// multiple types of storage in the same IPC mem block, so we must cache the
// device ptr to construct typed storage as it comes.
//
// ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the process
// that can be used to access the memory block in the sender process.
// It only saves a weak_ptr of the device pointer in the map, the shared_ptr
// will be used to reconstruct all storages in this CudaMalloc allocation.
// And it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
//
namespace {
  std::mutex IpcMutex;
  std::unordered_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
}

std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  std::lock_guard<std::mutex> lock(IpcMutex);

  auto iter = ipcMemHandle_to_devptr.find(handle);
  if (iter != ipcMemHandle_to_devptr.end()) {
    auto devptr = iter->second.lock();
    if (devptr) return devptr;
  }
  // This ipcMemHandle hasn't been opened, or already expired, open it to
  // enable IPC access to that mem block.
  void *dev = nullptr;
  auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  // devPtr has to be deleted in same device when created.
  int curr_device;
  C10_CUDA_CHECK(cudaGetDevice(&curr_device));
  auto sp = std::shared_ptr<void>(
      dev,
      [handle, curr_device](void *ptr) {
        cuda::CUDAGuard device_guard(curr_device);
        std::lock_guard<std::mutex> deleter_lock(IpcMutex);
        C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
        ipcMemHandle_to_devptr.erase(handle);});
  std::weak_ptr<void> wp = sp;
  // To eliminate an additional search, we can use insert().
  // It doesn't overwrite when key already exists(ptr expired).
  // But in the deleter for sp we erased the entry,
  // this should be safe to do now.
  ipcMemHandle_to_devptr.insert(iter, {handle, wp});

  return sp;
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
  // printf("device %d at %p with %lu\n", device, r, nbytes);
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}

void renew_shared_memory() {
  caching_allocator.renew_shared_memory();
}

void mark_shared_memory() {
  caching_allocator.mark_shared_memory();
}

void unmark_shared_memory() {
  caching_allocator.unmark_shared_memory();
}

} // namespace CUDACachingAllocator

}} // namespace c10::cuda
