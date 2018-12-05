#include "utils.h"
#include "objdetect_prv.h"

static objdetect_struct* objdet_info = NULL;

void objdetect_free
    (
    void
    )
{
#ifdef NNPACK
    pthreadpool_destroy(objdet_info->net.threadpool);
    nnp_deinitialize();
#endif
}

// comment below is the value you would like to take care and aware of ...
void objdetect_init
    (
    char* weight_file_path,
    const int netw,
    const int neth
    )
{
    objdet_info = (objdetect_struct*)alloc_from_stack(sizeof(objdetect_struct));
    objdet_info->net.n = 86;
    objdet_info->net.layers = (layer_struct*)alloc_from_stack(objdet_info->net.n * sizeof(layer_struct));
    objdet_info->net.w = netw;
    objdet_info->net.h = neth;
    objdet_info->net.c = 3;
    objdet_info->class_num = 1; // person only
    objdet_info->thresh = 0.35f;
    objdet_info->nms_thresh = 0.45f;

    FILE *fp = fopen(weight_file_path, "rb");
    if(!fp)
        {
        printf("failed to open weight file\n");
        exit(EXIT_FAILURE);
        }
    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2)
        {
        size_t iseen = 0;
        fread(&iseen, sizeof(size_t), 1, fp);
        }
    else
        {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        }

    layer_struct *layer_ptr = NULL;
    layer_struct *prev_layer_ptr = NULL;
    // layer 0 conv0
    layer_ptr = objdet_info->net.layers;
    layer_ptr->n = 32;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = objdet_info->net.w;
    layer_ptr->h = objdet_info->net.h;
    layer_ptr->c = objdet_info->net.c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    int num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 1 conv1/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 2 conv1
    layer_ptr->n = 64;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 3 conv2/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 4 conv2
    layer_ptr->n = 128;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 5 conv3/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 6 conv3
    layer_ptr->n = 128;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 7 conv4/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 8 conv4
    layer_ptr->n = 256;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 9 conv5/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 10 conv5
    layer_ptr->n = 256;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 11 conv6/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 12 conv6
    layer_ptr->n = 512;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 13 conv7/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 14 conv7
    layer_ptr->n = 512;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 15 conv8/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 16 conv8
    layer_ptr->n = 512;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 17 conv9/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 18 conv9
    layer_ptr->n = 512;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 19 conv10/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 20 conv10
    layer_ptr->n = 512;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 21 conv11/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 22 conv11
    layer_ptr->n = 512;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 23 conv12/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 24 conv12
    layer_ptr->n = 1024;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 25 conv13/dw
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(prev_layer_ptr->out_c * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_group_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), prev_layer_ptr->out_c, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 26 conv13
    layer_ptr->n = 1024;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 27 conv14_1
    layer_ptr->n = 256;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 28 conv14_2
    layer_ptr->n = 512;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 29 conv15_1
    layer_ptr->n = 128;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 30 conv15_2
    layer_ptr->n = 256;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 31 conv16_1
    layer_ptr->n = 128;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 32 conv16_2
    layer_ptr->n = 256;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 33 conv17_1
    layer_ptr->n = 64;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 34 conv17_2
    layer_ptr->n = 128;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = relu_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 35 route from conv11
    layer_ptr->route_index = 22;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 36 conv11_mbox_loc
    layer_ptr->n = 12;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 37 conv11_mbox_loc_perm & conv11_mbox_loc_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 38 route from conv11
    layer_ptr->route_index = 22;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 39 conv11_mbox_conf
    layer_ptr->n = 3 * (objdet_info->class_num + 1);
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 40 conv11_mbox_conf_perm & conv11_mbox_conf_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 41 route from conv11
    layer_ptr->route_index = 22;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 42 conv11_mbox_priorbox
    layer_ptr->min_size = 60.0f;
    layer_ptr->max_size = 0.0f;
    layer_ptr->aspect_ratio_num = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    if (layer_ptr->max_size > 0.0f)
        {
        layer_ptr->c = 1 + 1 + layer_ptr->aspect_ratio_num;
        }
    else
        {
        layer_ptr->c = 1 + layer_ptr->aspect_ratio_num;
        }
    layer_ptr->outputs = 2 * 4 * layer_ptr->w * layer_ptr->h * layer_ptr->c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_priorbox_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 43 route from conv13
    layer_ptr->route_index = 26;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 44 conv13_mbox_loc
    layer_ptr->n = 24;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 45 conv13_mbox_loc_perm & conv13_mbox_loc_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 46 route from conv13
    layer_ptr->route_index = 26;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 47 conv13_mbox_conf
    layer_ptr->n = 6 * (objdet_info->class_num + 1);
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 48 conv13_mbox_conf_perm & conv13_mbox_conf_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 49 route from conv13
    layer_ptr->route_index = 26;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 50 conv13_mbox_priorbox
    layer_ptr->min_size = 105.0f;
    layer_ptr->max_size = 150.0f;
    layer_ptr->aspect_ratio_num = 4;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    if (layer_ptr->max_size > 0.0f)
        {
        layer_ptr->c = 1 + 1 + layer_ptr->aspect_ratio_num;
        }
    else
        {
        layer_ptr->c = 1 + layer_ptr->aspect_ratio_num;
        }
    layer_ptr->outputs = 2 * 4 * layer_ptr->w * layer_ptr->h * layer_ptr->c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_priorbox_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 51 route from conv14_2
    layer_ptr->route_index = 28;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 52 conv14_2_mbox_loc
    layer_ptr->n = 24;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 53 conv14_2_mbox_loc_perm & conv14_2_mbox_loc_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 54 route from conv14_2
    layer_ptr->route_index = 28;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 55 conv14_2_mbox_conf
    layer_ptr->n = 6 * (objdet_info->class_num + 1);
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 56 conv14_2_mbox_conf_perm & conv14_2_mbox_conf_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 57 route from conv14_2
    layer_ptr->route_index = 28;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 58 conv14_2_mbox_priorbox
    layer_ptr->min_size = 150.0f;
    layer_ptr->max_size = 195.0f;
    layer_ptr->aspect_ratio_num = 4;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    if (layer_ptr->max_size > 0.0f)
        {
        layer_ptr->c = 1 + 1 + layer_ptr->aspect_ratio_num;
        }
    else
        {
        layer_ptr->c = 1 + layer_ptr->aspect_ratio_num;
        }
    layer_ptr->outputs = 2 * 4 * layer_ptr->w * layer_ptr->h * layer_ptr->c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_priorbox_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 59 route from conv15_2
    layer_ptr->route_index = 30;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 60 conv15_2_mbox_loc
    layer_ptr->n = 24;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 61 conv15_2_mbox_loc_perm & conv15_2_mbox_loc_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 62 route from conv15_2
    layer_ptr->route_index = 30;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 63 conv15_2_mbox_conf
    layer_ptr->n = 6 * (objdet_info->class_num + 1);
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 64 conv15_2_mbox_conf_perm & conv15_2_mbox_conf_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 65 route from conv15_2
    layer_ptr->route_index = 30;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 66 conv15_2_mbox_priorbox
    layer_ptr->min_size = 195.0f;
    layer_ptr->max_size = 240.0f;
    layer_ptr->aspect_ratio_num = 4;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    if (layer_ptr->max_size > 0.0f)
        {
        layer_ptr->c = 1 + 1 + layer_ptr->aspect_ratio_num;
        }
    else
        {
        layer_ptr->c = 1 + layer_ptr->aspect_ratio_num;
        }
    layer_ptr->outputs = 2 * 4 * layer_ptr->w * layer_ptr->h * layer_ptr->c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_priorbox_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 67 route from conv16_2
    layer_ptr->route_index = 32;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 68 conv16_2_mbox_loc
    layer_ptr->n = 24;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 69 conv16_2_mbox_loc_perm & conv16_2_mbox_loc_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 70 route from conv16_2
    layer_ptr->route_index = 32;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 71 conv16_2_mbox_conf
    layer_ptr->n = 6 * (objdet_info->class_num + 1);
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 72 conv16_2_mbox_conf_perm & conv16_2_mbox_conf_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 73 route from conv16_2
    layer_ptr->route_index = 32;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 74 conv16_2_mbox_priorbox
    layer_ptr->min_size = 240.0f;
    layer_ptr->max_size = 285.0f;
    layer_ptr->aspect_ratio_num = 4;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    if (layer_ptr->max_size > 0.0f)
        {
        layer_ptr->c = 1 + 1 + layer_ptr->aspect_ratio_num;
        }
    else
        {
        layer_ptr->c = 1 + layer_ptr->aspect_ratio_num;
        }
    layer_ptr->outputs = 2 * 4 * layer_ptr->w * layer_ptr->h * layer_ptr->c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_priorbox_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 75 route from conv17_2
    layer_ptr->route_index = 34;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 76 conv17_2_mbox_loc
    layer_ptr->n = 24;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 77 conv17_2_mbox_loc_perm & conv17_2_mbox_loc_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 78 route from conv17_2
    layer_ptr->route_index = 34;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 79 conv17_2_mbox_conf
    layer_ptr->n = 6 * (objdet_info->class_num + 1);
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = (layer_ptr->w + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_h = (layer_ptr->h + 2*layer_ptr->pad - layer_ptr->size) / layer_ptr->stride + 1;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer_linear;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 80 conv17_2_mbox_conf_perm & conv17_2_mbox_conf_flat
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_permute_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 81 route from conv17_2
    layer_ptr->route_index = 34;
    prev_layer_ptr = objdet_info->net.layers + layer_ptr->route_index;
    layer_ptr->out_w = prev_layer_ptr->out_w;
    layer_ptr->out_h = prev_layer_ptr->out_h;
    layer_ptr->out_c = prev_layer_ptr->out_c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_route_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 82 conv17_2_mbox_priorbox
    layer_ptr->min_size = 285.0f;
    layer_ptr->max_size = 300.0f;
    layer_ptr->aspect_ratio_num = 4;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    if (layer_ptr->max_size > 0.0f)
        {
        layer_ptr->c = 1 + 1 + layer_ptr->aspect_ratio_num;
        }
    else
        {
        layer_ptr->c = 1 + layer_ptr->aspect_ratio_num;
        }
    layer_ptr->outputs = 2 * 4 * layer_ptr->w * layer_ptr->h * layer_ptr->c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_priorbox_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;
    // layer 83 mbox_loc
    layer_ptr->concat_index[0] = 6;
    layer_ptr->concat_index[1] = 37; // conv11_mbox_loc_flat
    layer_ptr->concat_index[2] = 45; // conv13_mbox_loc_flat
    layer_ptr->concat_index[3] = 53; // conv14_2_mbox_loc_flat
    layer_ptr->concat_index[4] = 61; // conv15_2_mbox_loc_flat
    layer_ptr->concat_index[5] = 69; // conv16_2_mbox_loc_flat
    layer_ptr->concat_index[6] = 77; // conv17_2_mbox_loc_flat
    int i = 0;
    for (i = 0; i < layer_ptr->concat_index[0]; ++i)
        {
        layer_ptr->outputs += objdet_info->net.layers[layer_ptr->concat_index[i+1]].outputs;
        }
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_concat_1d_layer;
    ++layer_ptr;
    // layer 84 mbox_conf
    layer_ptr->concat_index[0] = 6;
    layer_ptr->concat_index[1] = 40; // conv11_mbox_conf_flat
    layer_ptr->concat_index[2] = 48; // conv13_mbox_conf_flat
    layer_ptr->concat_index[3] = 56; // conv14_2_mbox_conf_flat
    layer_ptr->concat_index[4] = 64; // conv15_2_mbox_conf_flat
    layer_ptr->concat_index[5] = 72; // conv16_2_mbox_conf_flat
    layer_ptr->concat_index[6] = 80; // conv17_2_mbox_conf_flat
    for (i = 0; i < layer_ptr->concat_index[0]; ++i)
        {
        layer_ptr->outputs += objdet_info->net.layers[layer_ptr->concat_index[i+1]].outputs;
        }
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_concat_1d_layer;
    ++layer_ptr;
    // layer 85 mbox_priorbox
    layer_ptr->concat_index[0] = 6;
    layer_ptr->concat_index[1] = 42; // conv11_mbox_priorbox
    layer_ptr->concat_index[2] = 50; // conv13_mbox_priorbox
    layer_ptr->concat_index[3] = 58; // conv14_2_mbox_priorbox
    layer_ptr->concat_index[4] = 66; // conv15_2_mbox_priorbox
    layer_ptr->concat_index[5] = 74; // conv16_2_mbox_priorbox
    layer_ptr->concat_index[6] = 82; // conv17_2_mbox_priorbox
    for (i = 0; i < layer_ptr->concat_index[0]; ++i)
        {
        layer_ptr->outputs += objdet_info->net.layers[layer_ptr->concat_index[i+1]].outputs;
        }
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_concat_2d_layer;
    ++layer_ptr;


    fclose(fp);

#ifdef NNPACK
    nnp_initialize();
    objdet_info->net.threadpool = pthreadpool_create(4);
#endif
}

int *objdetect_main
    (
    unsigned char *im,
    int imw,
    int imh
    )
{
    unsigned int marksize = get_stack_current_alloc_size();
    objdet_info->src = im;
    objdet_info->srcw = imw;
    objdet_info->srch = imh;
    // split and convert to float32
    objdet_info->net.input = preprocessed(objdet_info);
    // set zero number of people detected
    objdet_info->output[0] = 0;

    //double time = what_time_is_it_now();
    network_predict(objdet_info->net);
    //printf("Predicted in %f seconds.\n", what_time_is_it_now() - time);

    get_detection_out(objdet_info);
    //clear_network(objdet_info->net);
    reset_stack_ptr_to_assigned_position(marksize);
    return objdet_info->output;
}

void set_objdetect_parameter
    (
    const float in_th
    )
{
    objdet_info->thresh = in_th;
}
