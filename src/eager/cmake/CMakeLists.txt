

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
  cmake_policy(SET CMP0117 NEW)
endif()

message(${PYTHON_EXECUTABLE})
if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)


# Project
project(torch_ort C CXX)

set(SELF_DIR "/bert_ort/chenta/vsts/ort/src/eager")
set(REPO_ROOT_DIR ${SELF_DIR}/../..)

set(ORT_SRC_DIR ${REPO_ROOT_DIR}/external/onnxruntime)

set(ORT_BUILD_DIR ${SELF_DIR}/ort_build/Debug)

set(ORT_PYTHON_BIND_PATH ${ORT_SRC_DIR}/onnxruntime/python)
set(ORT_PYTHON_BIND_TRAINING_PATH ${ORT_SRC_DIR}/orttraining/orttraining/python)

set(ORT_INCLUDE_DIRS ${ORT_SRC_DIR}/include/onnxruntime
    ${ORT_SRC_DIR}/include/onnxruntime/core/session
    ${ORT_SRC_DIR}/onnxruntime
    ${ORT_SRC_DIR}/orttraining
    ${ORT_BUILD_DIR}
    ${ORT_SRC_DIR}/cmake/external/onnx
    ${ORT_SRC_DIR}/cmake/external/SafeInt
    ${ORT_SRC_DIR}/cmake/external/protobuf/src
    ${ORT_SRC_DIR}/cmake/external/nsync/public
    ${ORT_SRC_DIR}/cmake/external/mp11/include
    ${ORT_SRC_DIR}/cmake/external/optional-lite/include
    ${ORT_SRC_DIR}/cmake/external/dlpack/include
)
list(APPEND ORT_INCLUDE_DIRS ${ORT_BUILD_DIR}/external/onnx)
# How to include np.get_include()??
message(${ORT_INCLUDE_DIRS})

set(ORT_STATIC_LIBS ${ORT_BUILD_DIR}/libonnxruntime_eager.a
    ${ORT_BUILD_DIR}/libonnxruntime_training.a
    ${ORT_BUILD_DIR}/libonnxruntime_session.a
    ${ORT_BUILD_DIR}/libonnxruntime_providers.a
    ${ORT_BUILD_DIR}/libonnxruntime_framework.a
    ${ORT_BUILD_DIR}/libonnxruntime_optimizer.a
    ${ORT_BUILD_DIR}/libonnxruntime_util.a
    ${ORT_BUILD_DIR}/libonnxruntime_graph.a
    ${ORT_BUILD_DIR}/libonnxruntime_mlas.a
    ${ORT_BUILD_DIR}/libonnxruntime_flatbuffers.a
    ${ORT_BUILD_DIR}/libonnxruntime_common.a
    ${ORT_BUILD_DIR}/libonnxruntime_optimizer.a
    ${ORT_BUILD_DIR}/external/nsync/libnsync_cpp.a
    ${ORT_BUILD_DIR}/external/onnx/libonnx.a
    ${ORT_BUILD_DIR}/external/onnx/libonnx_proto.a
    ${ORT_BUILD_DIR}/external/protobuf/cmake/libprotobufd.a
    ${ORT_BUILD_DIR}/external/re2/libre2.a
    ${ORT_BUILD_DIR}/tensorboard/libtensorboard.a
)

file(GLOB EAGER_SRC ${SELF_DIR}/*.cpp
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_exceptions.cc
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_iobinding.cc
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_mlvalue.cc
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_ortvalue.cc
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_state_common.cc
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_state.cc
    ${ORT_PYTHON_BIND_PATH}/onnxruntime_pybind_sparse_tensor.cc
    ${ORT_PYTHON_BIND_TRAINING_PATH}/orttraining_pybind_state.cc
    ${ORT_SRC_DIR}/orttraining/orttraining/core/framework/torch/dlpack_python.cc
)

list(APPEND CMAKE_PREFIX_PATH "/bert_ort/chenta/vsts/ort/external/pytorch/torch/")
find_package(Torch REQUIRED)
set(pybind11_DIR "/data/anaconda/envs/eager-ort/lib/python3.6/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

message(${TORCH_LIBRARIES})
list(APPEND TORCH_LIBRARIES "/bert_ort/chenta/vsts/ort/external/pytorch/torch/lib/libtorch_python.so")

add_library(torch_ort SHARED ${EAGER_SRC})
target_include_directories(torch_ort PRIVATE ${ORT_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_compile_definitions(torch_ort PRIVATE -DONNX_ML -DONNX_NAMESPACE=onnx -DENABLE_TRAINING -DONNX_BUILD_CONFIG=Debug)
target_link_libraries(torch_ort ${ORT_STATIC_LIBS} ${TORCH_LIBRARIES} pybind11::module)
set_property(TARGET torch_ort APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker -rpath=\\$ORIGIN")