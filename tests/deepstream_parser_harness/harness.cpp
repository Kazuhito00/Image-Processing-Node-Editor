/*
 * Test driver for the generated nvinfer bbox parser.
 *
 *   harness <yolo|yolox> <net_w> <net_h> <num_classes> <threshold>
 *           <tensor.bin> <dim0> [dim1 ...]
 *
 * Reads a raw float32 tensor, runs the parser like nvinfer would and prints
 * one "classId left top width height confidence" line per object.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCvStudioYolo(
    std::vector<NvDsInferLayerInfo> const &, NvDsInferNetworkInfo const &,
    NvDsInferParseDetectionParams const &,
    std::vector<NvDsInferObjectDetectionInfo> &);
extern "C" bool NvDsInferParseCvStudioYoloX(
    std::vector<NvDsInferLayerInfo> const &, NvDsInferNetworkInfo const &,
    NvDsInferParseDetectionParams const &,
    std::vector<NvDsInferObjectDetectionInfo> &);

int main(int argc, char **argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: see file header\n");
    return 2;
  }
  NvDsInferNetworkInfo network;
  network.width = std::atoi(argv[2]);
  network.height = std::atoi(argv[3]);
  network.channels = 3;

  NvDsInferParseDetectionParams params;
  params.numClassesConfigured = std::atoi(argv[4]);
  params.perClassPreclusterThreshold.assign(params.numClassesConfigured,
                                            std::atof(argv[5]));

  NvDsInferLayerInfo layer;
  std::memset(&layer, 0, sizeof(layer));
  layer.dataType = FLOAT;
  layer.layerName = "output";
  layer.inferDims.numElements = 1;
  for (int i = 7; i < argc; ++i) {
    const unsigned int dim = std::atoi(argv[i]);
    layer.inferDims.d[layer.inferDims.numDims++] = dim;
    layer.inferDims.numElements *= dim;
  }

  std::vector<float> data(layer.inferDims.numElements);
  std::FILE *file = std::fopen(argv[6], "rb");
  if (!file ||
      std::fread(data.data(), sizeof(float), data.size(), file) != data.size()) {
    std::fprintf(stderr, "cannot read %zu floats from %s\n", data.size(), argv[6]);
    return 2;
  }
  std::fclose(file);
  layer.buffer = data.data();

  std::vector<NvDsInferLayerInfo> layers(1, layer);
  std::vector<NvDsInferObjectDetectionInfo> objects;
  const bool ok = std::strcmp(argv[1], "yolox") == 0
      ? NvDsInferParseCvStudioYoloX(layers, network, params, objects)
      : NvDsInferParseCvStudioYolo(layers, network, params, objects);
  if (!ok) {
    return 1;
  }
  for (const NvDsInferObjectDetectionInfo &o : objects) {
    std::printf("%u %.4f %.4f %.4f %.4f %.6f\n", o.classId, o.left, o.top,
                o.width, o.height, o.detectionConfidence);
  }
  return 0;
}
