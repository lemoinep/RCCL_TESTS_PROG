#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas-export.h"
#include "hipblas.h"
#include "hipsolver.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <roctx.h> //Scan Perf

#include <rccl.h> // For multi-GPU

#include <random>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      printf("error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,    \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define RCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t result = call;                                                \
    if (result != ncclSuccess) {                                               \
      std::cerr << "RCCL error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << ncclGetErrorString(result) << std::endl;                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void onKernelNothing(float4 *nothing) {
  // nothing void
}

void runScanPreheatingGPU() {
  int nDevices;
  hipGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; ++i) {
    hipSetDevice(i);
    float4 *d_nothing;
    hipMalloc(&d_nothing, 100 * sizeof(float4));
    onKernelNothing<<<1, 1>>>(d_nothing);
    hipFree(d_nothing);
  }
}

void testRCCL() {
  ncclComm_t comm;
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  ncclCommInitRank(&comm, 1, id, 0);
  //...
  ncclCommDestroy(comm);
}

void testRCCL2() {
  std::cout << "[INFO]: Test RCCL\n";
  int nDevices;
  hipError_t err = hipGetDeviceCount(&nDevices);
  if (err != hipSuccess) {
    std::cerr << "Error getting device count: " << hipGetErrorString(err)
              << std::endl;
    exit(1);
  }

  std::vector<int> devices(nDevices);
  for (int i = 0; i < nDevices; ++i) {
    devices[i] = i;
  }

  // Initialization of RCCL communicators
  std::vector<ncclComm_t> comms(nDevices);
  ncclUniqueId id;
  RCCL_CHECK(ncclGetUniqueId(&id));

  // Parallel initialization of communicators
  RCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    err = hipSetDevice(i);
    if (err != hipSuccess) {
      std::cerr << "Error setting device " << i << ": "
                << hipGetErrorString(err) << std::endl;
      exit(1);
    }
    RCCL_CHECK(ncclCommInitRank(&comms[i], nDevices, id, i));
  }
  RCCL_CHECK(ncclGroupEnd());

  // Synchronisation
  for (int i = 0; i < nDevices; ++i) {
    err = hipSetDevice(i);
    if (err != hipSuccess) {
      std::cerr << "Error setting device " << i << ": "
                << hipGetErrorString(err) << std::endl;
      exit(1);
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      std::cerr << "Error synchronizing device " << i << ": "
                << hipGetErrorString(err) << std::endl;
      exit(1);
    }
  }

  // Cleaning
  for (int i = 0; i < nDevices; ++i) {
    RCCL_CHECK(ncclCommDestroy(comms[i]));
  }

  std::cout
      << "[INFO]: RCCL Communicators Initialized and Destroyed Successfully."
      << std::endl;
}

void testRCCL3() {
  std::cout << "[INFO]: Test RCCL avec streaming\n";
  int nDevices;
  HIP_CHECK(hipGetDeviceCount(&nDevices));

  std::vector<int> devices(nDevices);
  std::vector<hipStream_t> streams(nDevices);
  std::vector<float *> sendbuff(nDevices);
  std::vector<float *> recvbuff(nDevices);
  const int size = 32 * 1024 * 1024; // 32M éléments

  std::cout << "[INFO]: OK1." << std::endl;

  for (int i = 0; i < nDevices; ++i) {
    devices[i] = i;
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipStreamCreate(&streams[i]));
    HIP_CHECK(hipMalloc(&sendbuff[i], size * sizeof(float)));
    HIP_CHECK(hipMalloc(&recvbuff[i], size * sizeof(float)));
    HIP_CHECK(hipMemset(sendbuff[i], 1, size * sizeof(float)));
    HIP_CHECK(hipMemset(recvbuff[i], 0, size * sizeof(float)));
  }

  std::cout << "[INFO]: OK2." << std::endl;

  // Initialization of RCCL communicators
  std::vector<ncclComm_t> comms(nDevices);
  ncclUniqueId id;
  RCCL_CHECK(ncclGetUniqueId(&id));

  std::cout << "[INFO]: OK3." << std::endl;

  // Parallel initialization of communicators
  RCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    RCCL_CHECK(ncclCommInitRank(&comms[i], nDevices, id, i));
  }
  RCCL_CHECK(ncclGroupEnd());

  std::cout << "[INFO]: OK4." << std::endl;

  // Perform RCCL AllReduce operation
  RCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    RCCL_CHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i],
                             size, ncclFloat, ncclSum, comms[i], streams[i]));
  }
  RCCL_CHECK(ncclGroupEnd());

  std::cout << "[INFO]: OK5." << std::endl;

  // Synchronisation des streams : error
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    hipError_t status = hipStreamQuery(streams[i]);
    if (status == hipSuccess) {
      std::cout << "Stream " << i << " completed successfully." << std::endl;
    } else if (status == hipErrorNotReady) {
      std::cout << "Stream " << i << " is still running." << std::endl;
    } else {
      std::cerr << "Error in stream " << i << ": " << hipGetErrorString(status)
                << std::endl;
    }
  }

  /*
      // Synchronisation des streams : error
      for (int i = 0; i < nDevices; ++i) {
          HIP_CHECK(hipSetDevice(i));
          HIP_CHECK(hipStreamSynchronize(streams[i]));
      }
      */

  std::cout << "[INFO]: OK6." << std::endl;

  // Vérification (sur le premier device pour simplifier)
  HIP_CHECK(hipSetDevice(0));
  std::vector<float> host_buffer(size);
  HIP_CHECK(hipMemcpy(host_buffer.data(), recvbuff[0], size * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::cout << "[INFO]: OK7." << std::endl;

  bool correct = true;
  for (int i = 0; i < size; ++i) {
    if (host_buffer[i] != nDevices) {
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "[INFO]: RCCL AllReduce completed successfully." << std::endl;
  } else {
    std::cout << "[ERROR]: RCCL AllReduce failed." << std::endl;

    std::cout << "[INFO]: OK8." << std::endl;

    // Nettoyage
    for (int i = 0; i < nDevices; ++i) {
      // hipSetDevice(i)
      ncclCommDestroy(comms[i]);
      HIP_CHECK(hipFree(sendbuff[i]));
      HIP_CHECK(hipFree(recvbuff[i]));
      HIP_CHECK(hipStreamDestroy(streams[i]));
    }

    std::cout << "[INFO]: RCCL test completed." << std::endl;
  }
}

void testRCCL4() {
  std::cout
      << "[INFO]: RCCL test with vector fragmentation and reconstruction\n";

  // Initialization of devices
  int nDevices;
  hipError_t err = hipGetDeviceCount(&nDevices);
  if (err != hipSuccess) {
    std::cerr << "Error getting number of devices: " << hipGetErrorString(err)
              << std::endl;
    exit(1);
  }

  // Creation of the initial vector
  const int vectorSize = 1000000;
  std::vector<float> initialVector(vectorSize);
  for (int i = 0; i < vectorSize; ++i) {
    initialVector[i] = static_cast<float>(i);
  }

  // Calculation of fragment size
  int fragmentSize = vectorSize / nDevices;
  int lastFragmentSize = fragmentSize + (vectorSize % nDevices);

  // Initialization of RCCL communicators
  std::vector<ncclComm_t> comms(nDevices);
  ncclUniqueId id;
  RCCL_CHECK(ncclGetUniqueId(&id));

  // Parallel initialization of communicators
  RCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    RCCL_CHECK(ncclCommInitRank(&comms[i], nDevices, id, i));
  }
  RCCL_CHECK(ncclGroupEnd());

  // Allocate and copy fragments on each GPU
  std::vector<float *> d_fragments(nDevices);
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    int currentFragmentSize =
        (i == nDevices - 1) ? lastFragmentSize : fragmentSize;
    HIP_CHECK(hipMalloc(&d_fragments[i], currentFragmentSize * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_fragments[i], initialVector.data() + i * fragmentSize,
                        currentFragmentSize * sizeof(float),
                        hipMemcpyHostToDevice));
  }

  // Synchronisation
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
  }

  //****

  // Operations on fragments
  std::cout << "[INFO]: Multiplying each element by 2 on GPUs\n";

  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    int currentFragmentSize =
        (i == nDevices - 1) ? lastFragmentSize : fragmentSize;

    auto multiplyBy2 = [=] __host__ __device__(int idx) {
      if (idx < currentFragmentSize) {
        d_fragments[i][idx] *= 2.0f;
      }
    };

    // Launching the kernel
    const int threadsPerBlock = 256;
    const int blocks =
        (currentFragmentSize + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(multiplyBy2, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       currentFragmentSize);

    // Error checking after kernel launch
    HIP_CHECK(hipGetLastError());
  }

  // Synchronization after operations
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
  }

  //****

  // Vector reconstruction
  std::vector<float> reconstructedVector(vectorSize);
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    int currentFragmentSize =
        (i == nDevices - 1) ? lastFragmentSize : fragmentSize;
    HIP_CHECK(hipMemcpy(reconstructedVector.data() + i * fragmentSize,
                        d_fragments[i], currentFragmentSize * sizeof(float),
                        hipMemcpyDeviceToHost));
  }

  // Verification
  bool correct = true;
  for (int i = 0; i < vectorSize; ++i) {
    if (initialVector[i] != reconstructedVector[i]) {
      correct = false;
      std::cout << "Mismatch at index " << i << ": " << reconstructedVector[i]
                << " != " << initialVector[i] * 2.0f << std::endl;
      break;
    }
  }

  std::cout << "[INFO]: Vector reconstructionr "
            << (correct ? "successful" : "failed") << std::endl;

  // Displaying results
  const int displayCount =
      10; // Number of items to display at the start and end

  std::cout << "\n[INFO]: Displaying results\n";

  // Display the initial vector
  std::cout << "Reconstructed vector (first " << displayCount
            << " elements) :\n";
  for (int i = 0; i < displayCount && i < vectorSize; ++i) {
    std::cout << std::setw(10) << std::fixed << std::setprecision(2)
              << initialVector[i] << " ";
  }
  std::cout << "...\n";

  if (vectorSize > displayCount * 2) {
    std::cout << "Reconstructed vector (latest " << displayCount
              << " elements) :\n";
    for (int i = vectorSize - displayCount; i < vectorSize; ++i) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                << initialVector[i] << " ";
    }
    std::cout << "\n";
  }

  // Display the reconstructed vector
  std::cout << "\nReconstructed vector (first " << displayCount
            << " elements) :\n";
  for (int i = 0; i < displayCount && i < vectorSize; ++i) {
    std::cout << std::setw(10) << std::fixed << std::setprecision(2)
              << reconstructedVector[i] << " ";
  }
  std::cout << "...\n";

  if (vectorSize > displayCount * 2) {
    std::cout << "Reconstructed vector (latest " << displayCount
              << " elements) :\n";
    for (int i = vectorSize - displayCount; i < vectorSize; ++i) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                << reconstructedVector[i] << " ";
    }
    std::cout << "\n";
  }

  // Display some statistics
  double sumInitial = 0, sumReconstructed = 0;
  for (int i = 0; i < vectorSize; ++i) {
    sumInitial += initialVector[i];
    sumReconstructed += reconstructedVector[i];
  }

  std::cout << "\nSum of the elements of the initial vector : " << std::fixed
            << std::setprecision(2) << sumInitial << std::endl;
  std::cout << "Sum of the elements of the reconstructed vector : "
            << std::fixed << std::setprecision(2) << sumReconstructed
            << std::endl;
  std::cout << "Report (reconstructed / initial): " << std::fixed
            << std::setprecision(4) << (sumReconstructed / sumInitial)
            << std::endl;

  // Nettoyage
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipFree(d_fragments[i]));
    RCCL_CHECK(ncclCommDestroy(comms[i]));
  }

  std::cout << "[INFO]: RCCL test completed successfully." << std::endl;
}

void testRCCL5() {
  std::cout << "[INFO]: RCCL test with vector fragmentation, reconstruction, "
               "and collective operations\n";

  // Initialization of devices
  int nDevices;
  HIP_CHECK(hipGetDeviceCount(&nDevices));
  std::cout << "[INFO]: Number of devices: " << nDevices << std::endl;

  // Creation of the initial vector
  const int vectorSize = 300;
  std::vector<float> initialVector(vectorSize);
  for (int i = 0; i < vectorSize; ++i) {
    initialVector[i] = static_cast<float>(i);
  }
  std::cout << "[INFO]: Initial vector created." << std::endl;

  // Calculation of fragment size
  int fragmentSize = vectorSize / nDevices;
  int lastFragmentSize = fragmentSize + (vectorSize % nDevices);

  // Initialization of RCCL communicators
  std::vector<ncclComm_t> comms(nDevices);
  ncclUniqueId id;
  RCCL_CHECK(ncclGetUniqueId(&id));

  // Parallel initialization of communicators
  RCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    RCCL_CHECK(ncclCommInitRank(&comms[i], nDevices, id, i));
  }
  RCCL_CHECK(ncclGroupEnd());
  std::cout << "[INFO]: RCCL communicators initialized." << std::endl;

  // Allocate and copy fragments on each GPU
  std::vector<float *> d_fragments(nDevices);
  std::vector<float *> d_results(nDevices);
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    int currentFragmentSize =
        (i == nDevices - 1) ? lastFragmentSize : fragmentSize;
    HIP_CHECK(hipMalloc(&d_fragments[i], currentFragmentSize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_results[i], currentFragmentSize * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_fragments[i], initialVector.data() + i * fragmentSize,
                        currentFragmentSize * sizeof(float),
                        hipMemcpyHostToDevice));
  }
  std::cout << "[INFO]: Fragments allocated and copied to GPUs." << std::endl;

  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Syncronisation " << i << std::endl;
  }

  // Collective operations
  RCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDevices; ++i) {
    // HIP_CHECK(hipSetDevice(i));
    int currentFragmentSize =
        (i == nDevices - 1) ? lastFragmentSize : fragmentSize;
    RCCL_CHECK(ncclAllReduce(d_fragments[i], d_results[i], currentFragmentSize,
                             ncclFloat, ncclSum, comms[i], nullptr));
  }
  RCCL_CHECK(ncclGroupEnd());
  std::cout << "[INFO]: AllReduce operation completed." << std::endl;

  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    size_t free, total;
    HIP_CHECK(hipMemGetInfo(&free, &total));
    std::cout << "Device " << i << " memory: " << free << " / " << total
              << " bytes free" << std::endl;
  }

  /*
      for (int i = 0; i < nDevices; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipDeviceSynchronize());
      }
      std::cout << "[INFO]: Synchronisation completed." << std::endl;
  */

  // Vector reconstruction
  std::vector<float> reconstructedVector(vectorSize);
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    int currentFragmentSize =
        (i == nDevices - 1) ? lastFragmentSize : fragmentSize;
    HIP_CHECK(hipMemcpy(reconstructedVector.data() + i * fragmentSize,
                        d_results[i], currentFragmentSize * sizeof(float),
                        hipMemcpyDeviceToHost));
  }

  /*
   std::vector<float> reconstructedVector(vectorSize);
   std::vector<hipStream_t> streams(nDevices);
   for (int i = 0; i < nDevices; ++i) {
       HIP_CHECK(hipSetDevice(i));
       HIP_CHECK(hipStreamCreate(&streams[i]));
       int currentFragmentSize = (i == nDevices - 1) ? lastFragmentSize :
   fragmentSize; HIP_CHECK(hipMemcpyAsync(reconstructedVector.data() + i *
   fragmentSize, d_results[i], currentFragmentSize * sizeof(float),
   hipMemcpyDeviceToHost, streams[i]));
   }
   for (int i = 0; i < nDevices; ++i) {
       HIP_CHECK(hipSetDevice(i));
       HIP_CHECK(hipStreamSynchronize(streams[i]));
       HIP_CHECK(hipStreamDestroy(streams[i]));
   }
*/

  std::cout << "[INFO]: Vector reconstructed." << std::endl;

  // Verification
  bool correct = true;
  float expectedSum = nDevices * (vectorSize - 1) * vectorSize / 2.0f;
  for (int i = 0; i < vectorSize; ++i) {
    if (std::abs(reconstructedVector[i] - expectedSum) > 1e-5) {
      correct = false;
      std::cout << "Mismatch at index " << i << ": " << reconstructedVector[i]
                << " (expected " << expectedSum << ")" << std::endl;
      break;
    }
  }

  if (correct) {
    std::cout << "[INFO]: Verification passed. All values are correct."
              << std::endl;
  } else {
    std::cout << "[ERROR]: Verification failed. Some values are incorrect."
              << std::endl;
  }

  // Cleanup
  for (int i = 0; i < nDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipFree(d_fragments[i]));
    HIP_CHECK(hipFree(d_results[i]));
    RCCL_CHECK(ncclCommDestroy(comms[i]));
  }

  std::cout << "[INFO]: RCCL test completed successfully." << std::endl;
}

int main(int argc, char *argv[]) {

  // testCheckOverlap();

  bool isPreheating = false;

  if (argc > 1)
    isPreheating = bool(atoi(argv[1]));
  // std::cout<<argv[1];

  std::cout << "\n";
  if (isPreheating)
    // runPreheatingGPU(0);
    runScanPreheatingGPU();
  std::cout << "\n";

  // testRCCL2();
  // testRCCL3();
  testRCCL4();
  // testRCCL5();
  // testRCCL6();
  std::cout << "[INFO]: WELL DONE :-) FINISHED !\n";
  return 0;
}
