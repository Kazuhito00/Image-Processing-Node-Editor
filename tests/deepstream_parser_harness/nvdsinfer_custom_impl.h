/*
 * Minimal stand-in for the DeepStream SDK header of the same name, with the
 * nvinfer custom bbox parser API only. Lets the generated parser be compiled
 * and exercised on machines without the SDK.
 */
#ifndef CVSTUDIO_TEST_NVDSINFER_CUSTOM_IMPL_H
#define CVSTUDIO_TEST_NVDSINFER_CUSTOM_IMPL_H

#include <vector>

#define NVDSINFER_MAX_DIMS 8

typedef struct
{
  unsigned int numDims;
  unsigned int d[NVDSINFER_MAX_DIMS];
  unsigned int numElements;
} NvDsInferDims;

typedef enum
{
  FLOAT = 0,
  HALF = 1,
  INT8 = 2,
  INT32 = 3,
  INT64 = 4,
  UINT8 = 5
} NvDsInferDataType;

typedef struct
{
  NvDsInferDataType dataType;
  NvDsInferDims inferDims;
  int bindingIndex;
  const char *layerName;
  void *buffer;
  int isInput;
} NvDsInferLayerInfo;

typedef struct
{
  unsigned int width;
  unsigned int height;
  unsigned int channels;
} NvDsInferNetworkInfo;

typedef struct
{
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
  float rotation_angle;
} NvDsInferObjectDetectionInfo;

typedef struct
{
  unsigned int numClassesConfigured;
  std::vector<float> perClassPreclusterThreshold;
  std::vector<float> perClassPostclusterThreshold;
} NvDsInferParseDetectionParams;

typedef bool (*NvDsInferParseCustomFunc)(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
  static void checkFunc_##customParseFunc(NvDsInferParseCustomFunc func = customParseFunc) \
  { checkFunc_##customParseFunc(); }; \
  extern "C" bool customParseFunc( \
      std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
      NvDsInferNetworkInfo const &networkInfo, \
      NvDsInferParseDetectionParams const &detectionParams, \
      std::vector<NvDsInferObjectDetectionInfo> &objectList);

#endif
