#ifndef OBJDETECT_PRV_H
#define OBJDETECT_PRV_H

#ifdef NNPACK
#include <nnpack.h>
#endif

#include "utils.h"

//#define PERSON_IDX_IN_VOC 14
#define PERSON_IDX_IN_VOC 0

struct layerst;
typedef struct layerst layer_struct;
struct networkst;
typedef struct networkst network_struct;

typedef struct
    {
    int label;
    float score;
    float x;
    float y;
    float w;
    float h;
    }detobj_struct;

typedef struct
    {
    int index;
    int class_idx;
    float **probs;
    }sortable_bbox;

struct layerst
    {
    void (*activation)
        (
        float*,
        int
        );
    void (*forward)
        (
        struct layerst,
        struct networkst
        );
    int n; // filter numbers (output channels)
    int size; // filter size
    int stride; // sliding window jump step
    int pad; // extra area
    int h; // input height
    int w; // input width
    int c; // input channels
    int inputs; // total input pixels
    float *weights;
    float *biases;
    int out_h; // output height
    int out_w; // output width
    int out_c; // output channels
    int outputs; // total output pixels
    float *scales; // batch normalization coefficients
    float *rolling_mean; // batch normalization coefficients
    float *rolling_variance; // batch normalization coefficients
    float *output; // layer result
    int classes; // class number, for last region layer only
    int route_index; // for route layer only
    int concat_index[10]; // for concat layer only, the first element is concat num
    float min_size; // for priorbox layer only
    float max_size; // for priorbox layer only
    int aspect_ratio_num; // for priorbox layer only
    };

struct networkst
    {
    int n; // total layer number
    int h; // input image height
    int w; // input image width
    int c; // input image channels
    float *input; // input src (note it will be the buffer for every layer's input)
    layer_struct *layers; //pointer to each layer
    box_struct *result_boxes;
    float **result_probs;
#ifdef NNPACK
    pthreadpool_t threadpool;
#endif
    };

typedef struct
    {
    int class_num;
    float thresh;
    float nms_thresh;
    network_struct net;
    unsigned char* src;
    int srcw;
    int srch;
    int output[501]; // TODO: remove hard coded number here ([0]: count, [rest]: obj box)
    }objdetect_struct;

void forward_priorbox_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_permute_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_concat_2d_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_concat_1d_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_route_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_convolutional_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_group_convolutional_layer
    (
    layer_struct l,
    network_struct net
    );

void forward_convolutional_layer_linear
    (
    layer_struct l,
    network_struct net
    );

void logistic_activate
    (
    float *x,
    int num
    );

void relu_activate
    (
    float *x,
    int num
    );

void linear_activate
    (
    float *x,
    int num
    );

void leaky_activate
    (
    float *x,
    int num
    );

float* preprocessed
    (
    objdetect_struct* objdet_wksp
    );

void network_predict
    (
    network_struct net
    );

void get_detection_out
    (
    objdetect_struct* objdet_wksp
    );

void clear_network
    (
    network_struct net
    );

#endif // OBJDETECT_PRV_H
