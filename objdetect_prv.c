#include "objdetect_prv.h"

float overlap
    (
    float x1,
    float w1,
    float x2,
    float w2
    )
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection
    (
    box_struct a,
    box_struct b
    )
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0)
        {
        return 0;
        }
    float area = w * h;
    return area;
}

float box_iou
    (
    box_struct a,
    box_struct b
    )
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return i / u;
}

void softmax
    (
    float *input,
    float *output,
    int stride, // l.w * l.h
    int num_class
    )
{
    int i = 0;
    int j = 0;
    float* inptr = input;
    float* outptr = output;
    for(i = 0; i < stride; ++i)
        {
        inptr = input + i;
        outptr = output + i;
        float sum = 0;
        float largest = -FLT_MAX;
        for(j = 0; j < num_class; ++j)
            {
            if(inptr[j * stride] > largest)
                {
                largest = inptr[j * stride];
                }
            }
        for(j = 0; j < num_class; ++j)
            {
            float e = exp(inptr[j * stride] - largest);
            sum += e;
            outptr[j * stride] = e;
            }
        for(j = 0; j < num_class; ++j)
            {
            outptr[j * stride] /= sum;
            }
        }
}

void logistic_activate
    (
    float *x,
    int num
    )
{
    int i = 0;
    for(i = 0; i < num; ++i)
        {
        x[i] = 1.0f / (1.0f + exp(-x[i]));
        }
}

void relu_activate
    (
    float *x,
    int num
    )
{
    int i = 0;
    for(i = 0; i < num; ++i)
        {
        if(x[i] < 0.0f)
            {
            x[i] = 0.0f;
            }
        }
}

void linear_activate
    (
    float *x,
    int num
    )
{
    return;
}

void leaky_activate
    (
    float *x,
    int num
    )
{
    int i = 0;
    for(i = 0; i < num; ++i)
        {
        if(x[i] < 0.0f)
            {
            x[i] = 0.1f * x[i];
            }
        }
}

void add_bias
    (
    float *output,
    float *biases,
    int n,
    int size
    )
{
    int i = 0;
    int j = 0;
    for(i = 0; i < n; ++i)
        {
        for(j = 0; j < size; ++j)
            {
            output[i * size + j] += biases[i];
            }
        }
}

void scale_bias
    (
    float *output,
    float *scales,
    int n,
    int size
    )
{
    int i = 0;
    int j = 0;
    for(i = 0; i < n; ++i)
        {
        for(j = 0; j < size; ++j)
            {
            output[i * size + j] *= scales[i];
            }
        }
}

void batch_normalize
    (
    float *src,
    float *mean,
    float *variance,
    int filters,
    int spatial
    )
{
    int f = 0;
    int i = 0;
    for(f = 0; f < filters; ++f)
        {
        for(i = 0; i < spatial; ++i)
            {
            int index = f * spatial + i;
            //TODO: deal with sqrt first
            src[index] = (src[index] - mean[f]) / (sqrt(variance[f]) + 0.000001f);
            }
        }
}

void gemm
    (
    int M,
    int N,
    int K,
    float *A,
    float *B,
    float *C
    )
{
    // TODO: try to use some blas library to speed up
    int i,j,k;
    //#pragma omp parallel for
    for(i = 0; i < M; ++i)
        {
        for(k = 0; k < K; ++k)
            {
            register float A_PART = A[i * K + k];
            for(j = 0; j < N; ++j)
                {
                C[i * N + j] += A_PART * B[k * N + j];
                }
            }
        }
}

float im2col_get_pixel
    (
    float *im,
    int height,
    int width,
    int row,
    int col,
    int channel,
    int pad
    )
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        {
        return 0;
        }
    return im[col + width*(row + height*channel)];
}

void im2col
    (
    const float *data_im,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int ksize,
    const int pad,
    const int stride,
    float *data_col
    )
{
    // flatten convolved area into one column
    int kernel_offset_c = 0;
    int kernel_offset_h = 0;
    int kernel_offset_w = 0;
    int in_offset_h = 0;
    int in_offset_w = 0;
    int dst_idx = 0;
    for (kernel_offset_c = 0; kernel_offset_c < in_channels; ++kernel_offset_c)
        {
        for (kernel_offset_h = 0; kernel_offset_h < ksize; ++kernel_offset_h)
            {
            for (kernel_offset_w = 0; kernel_offset_w < ksize; ++kernel_offset_w)
                {
                for (in_offset_h = 0; in_offset_h < in_height; in_offset_h += stride)
                    {
                    for (in_offset_w = 0; in_offset_w < in_width; in_offset_w += stride)
                        {
                        const int im_h_idx = kernel_offset_h + in_offset_h - pad;
                        const int im_w_idx = kernel_offset_w + in_offset_w - pad;
                        //const int kernel_idx = kernel_offset_c * ksize * ksize +
                                //kernel_offset_h * ksize + kernel_offset_w;
                        //const int dst_idx = kernel_idx * in_height * in_width +
                                //in_offset_h * in_width + in_offset_w;
                        if (im_h_idx < 0 || im_w_idx < 0 || im_h_idx >= in_height || im_w_idx >= in_width)
                            {
                            data_col[dst_idx] = 0.0f;
                            }
                        else
                            {
                            data_col[dst_idx] = data_im[kernel_offset_c * in_width * in_height +
                                                        im_h_idx * in_width + im_w_idx];
                            }
                        ++dst_idx;
                        }
                    }
                }
            }
        }
}

void forward_concat_2d_layer
    (
    layer_struct l,
    network_struct net
    )
{
    int i = 0;
    const int num_concat = l.concat_index[0];
    float *out = l.output;
    for (i = 0; i < num_concat; ++i)
        {
        float *in = net.layers[l.concat_index[i+1]].output;
        int insize = net.layers[l.concat_index[i+1]].outputs >> 1;
        memcpy(out, in, insize * sizeof(float));
        out += insize;
        }
    for (i = 0; i < num_concat; ++i)
        {
        float *in = net.layers[l.concat_index[i+1]].output;
        int insize = net.layers[l.concat_index[i+1]].outputs >> 1;
        memcpy(out, in + insize, insize * sizeof(float));
        out += insize;
        }
}

void forward_concat_1d_layer
    (
    layer_struct l,
    network_struct net
    )
{
    int i = 0;
    const int num_concat = l.concat_index[0];
    float *out = l.output;
    for (i = 0; i < num_concat; ++i)
        {
        float *in = net.layers[l.concat_index[i+1]].output;
        int insize = net.layers[l.concat_index[i+1]].outputs;
        memcpy(out, in, insize * sizeof(float));
        out += insize;
        }
}

void forward_route_layer
    (
    layer_struct l,
    network_struct net
    )
{
    float *in = net.layers[l.route_index].output;
    memcpy(l.output, in, l.outputs * sizeof(float));
}

void forward_group_convolutional_layer
    (
    layer_struct l,
    network_struct net
    )
{
#if NNPACK
    struct nnp_size input_size = { l.w, l.h };
    struct nnp_padding input_padding = { l.pad, l.pad, l.pad, l.pad };
    struct nnp_size kernel_size = { l.size, l.size };
    struct nnp_size stride = { l.stride, l.stride };
    const int filter_num = l.c;
    const int in_plane_size = l.w * l.h;
    const int out_plane_size = l.out_w * l.out_h;
    const int kernel_plane_size = l.size * l.size;
    int i = 0;
    for (i = 0; i < filter_num; ++i)
        {
        float *inptr = net.input + i * in_plane_size;
        float *wptr = l.weights + i * kernel_plane_size;
        float *bptr = l.biases + i;
        float *outptr = l.output + i * out_plane_size;
        nnp_convolution_inference
            (
            nnp_convolution_algorithm_implicit_gemm,
            nnp_convolution_transform_strategy_tuple_based,
            1,
            1,
            input_size,
            input_padding,
            kernel_size,
            stride,
            inptr,
            wptr,
            bptr,
            outptr,
            NULL,
            NULL,
            nnp_activation_relu,
            NULL,
            net.threadpool,
            NULL
            );
        }
#else
    const int NUM_FILTERS = l.c;
    const int IN_PLANE_SIZE = l.h * l.w;
    const int M = 1;
    const int N = l.out_w * l.out_h;
    const int K = l.size * l.size;
    int i = 0;
    for (i = 0; i < NUM_FILTERS; ++i)
        {
        float *intputptr = net.input + i * IN_PLANE_SIZE;
        float *weightptr = l.weights + i * K;
        float *rearrangedimptr = (float*)alloc_from_stack(N * K * sizeof(float));
        float *outptr = l.output + i * N;
        im2col(intputptr, 1, l.h, l.w, l.size, l.pad, l.stride, rearrangedimptr);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, weightptr, K, rearrangedimptr, N, 0.0f, outptr, N);
        partial_free_from_stack(N * K * sizeof(float));
        }
    add_bias(l.output, l.biases, l.out_c, N);
    l.activation(l.output, l.outputs);
#endif
}

void forward_convolutional_layer
    (
    layer_struct l,
    network_struct net
    )
{
#ifdef NNPACK
    struct nnp_size input_size = { l.w, l.h };
    struct nnp_padding input_padding = { l.pad, l.pad, l.pad, l.pad };
    struct nnp_size kernel_size = { l.size, l.size };
    struct nnp_size stride = { l.stride, l.stride };

    nnp_convolution_inference
        (
        nnp_convolution_algorithm_implicit_gemm,
        nnp_convolution_transform_strategy_tuple_based,
        l.c,
        l.n,
        input_size,
        input_padding,
        kernel_size,
        stride,
        net.input,
        l.weights,
        l.biases,
        l.output,
        NULL,
        NULL,
        nnp_activation_relu,
        NULL,
        net.threadpool,
        NULL
        );
#else
    /*
     *                                  ---------------
     *                                 /              / |
     *                                /              /  |
     *                               ---------------    |
     *                              |               |   |
     *                              |               |   |
     *                              |      in       |   |  ____
     *                              |               |   |      |
     *                              |               | /        |
     *                              |               |/         | im2col()
     *                               ---------------           |
     *                                                         |
     * //////////////////////////////////////////////////      |
     *                                                         |
     *                                         gemm()          |
     *                                                         v
     *                      weight length (K)        ------------------------------
     *                     -------------------      |                              |
     *                    |                   |     |                              |
     * num of filters (M) |                   |  *  |                              |
     *                    |                   |     |                              |
     *                     -------------------      |                              | weight length (K)
     *                                              |                              |
     *                                              |                              |
     *                                              |                              |
     *                                              |                              |
     *                                               ------------------------------
     *                                                    output plane size (N)
     */
    const int M = l.n;
    const int K = l.size * l.size * l.c;
    const int N = l.out_w * l.out_h;
    float *weightptr = l.weights;
    float *rearrangedimptr = (float*)alloc_from_stack(N * K * sizeof(float));
    float *outptr = l.output;
    if (l.size == 1)
        {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, weightptr, K, net.input, N, 0.0f, outptr, N);
        }
    else
        {
        im2col(net.input, l.c, l.h, l.w, l.size, l.pad, l.stride, rearrangedimptr);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, weightptr, K, rearrangedimptr, N, 0.0f, outptr, N);
        }
    add_bias(outptr, l.biases, l.out_c, N);
    l.activation(outptr, l.outputs);
    partial_free_from_stack(N * K * sizeof(float));
#endif
}

void forward_convolutional_layer_linear
    (
    layer_struct l,
    network_struct net
    )
{
#ifdef NNPACK
    struct nnp_size input_size = { l.w, l.h };
    struct nnp_padding input_padding = { l.pad, l.pad, l.pad, l.pad };
    struct nnp_size kernel_size = { l.size, l.size };
    struct nnp_size stride = { l.stride, l.stride };

    nnp_convolution_inference
        (
        nnp_convolution_algorithm_implicit_gemm,
        nnp_convolution_transform_strategy_tuple_based,
        l.c,
        l.n,
        input_size,
        input_padding,
        kernel_size,
        stride,
        net.input,
        l.weights,
        l.biases,
        l.output,
        NULL,
        NULL,
        nnp_activation_identity,
        NULL,
        net.threadpool,
        NULL
        );
#else
    const int M = l.n;
    const int K = l.size * l.size * l.c;
    const int N = l.out_w * l.out_h;
    float *weightptr = l.weights;
    float *rearrangedimptr = (float*)alloc_from_stack(N * K * sizeof(float));
    float *outptr = l.output;
    if (l.size == 1)
        {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, weightptr, K, net.input, N, 0.0f, outptr, N);
        }
    else
        {
        im2col(net.input, l.c, l.h, l.w, l.size, l.pad, l.stride, rearrangedimptr);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, weightptr, K, rearrangedimptr, N, 0.0f, outptr, N);
        }
    add_bias(outptr, l.biases, l.out_c, N);
    partial_free_from_stack(N * K * sizeof(float));
#endif
}

void forward_permute_layer
    (
    layer_struct l,
    network_struct net
    )
{
    float *in = net.input;
    float *out = l.output;
    const int H = l.out_h;
    const int W = l.out_w;
    const int C = l.out_c;
    const int HW = H * W;
    int i = 0;
    int j = 0;
    for (i = 0; i < HW; ++i)
        {
        float *inptr = in + i;
        float *outptr = out + i * C;
        for (j = 0; j < C; ++j)
            {
            outptr[j] = inptr[j * HW];
            }
        }
}

void forward_priorbox_layer
    (
    layer_struct l,
    network_struct net
    )
{
    const int layer_width = l.w;
    const int layer_height = l.h;
    const int img_width = net.w;
    const int img_height = net.h;
    const float step_w = (float)(img_width) / layer_width;
    const float step_h = (float)(img_height) / layer_height;
    const float min_size = l.min_size;
    const float max_size = l.max_size;
    const float sqrt_min_max_size = sqrt(min_size * max_size);
    const float offset = 0.5f;
    const float aspect_ratios[4] = { sqrt(2.0f), sqrt(1.0f / 2.0f),
                                     sqrt(3.0f), sqrt(1.0f / 3.0f) };
    const float variance[4] = {0.1f, 0.1f, 0.2f, 0.2f};
    const int ar_num = l.aspect_ratio_num;
    const int num_priors = l.c;
    float *top_data = l.output;
    int idx = 0;
    int h = 0;
    int w = 0;
    int r = 0;
    int i = 0;
    int j = 0;
    for (h = 0; h < layer_height; ++h)
        {
        for (w = 0; w < layer_width; ++w)
            {
            float center_x = (w + offset) * step_w;
            float center_y = (h + offset) * step_h;
            float box_width = min_size;
            float box_height = min_size;
            top_data[idx++] = (center_x - box_width / 2.) / img_width; // xmin
            top_data[idx++] = (center_y - box_height / 2.) / img_height; // ymin
            top_data[idx++] = (center_x + box_width / 2.) / img_width; // xmax
            top_data[idx++] = (center_y + box_height / 2.) / img_height; // ymax
            if (max_size > 0.0f)
                {
                box_width = sqrt_min_max_size;
                box_height = sqrt_min_max_size;
                top_data[idx++] = (center_x - box_width / 2.) / img_width; // xmin
                top_data[idx++] = (center_y - box_height / 2.) / img_height; // ymin
                top_data[idx++] = (center_x + box_width / 2.) / img_width; // xmax
                top_data[idx++] = (center_y + box_height / 2.) / img_height; // ymax
                }
            // rest of prior
            for (r = 0; r < ar_num; ++r)
                {
                float ar = aspect_ratios[r];
                box_width = min_size * ar;
                box_height = min_size / ar;
                top_data[idx++] = (center_x - box_width / 2.) / img_width; // xmin
                top_data[idx++] = (center_y - box_height / 2.) / img_height; // ymin
                top_data[idx++] = (center_x + box_width / 2.) / img_width; // xmax
                top_data[idx++] = (center_y + box_height / 2.) / img_height; // ymax
                }
            }
        }
    // set the variance
    for (h = 0; h < layer_height; ++h)
        {
        for (w = 0; w < layer_width; ++w)
            {
            for (i = 0; i < num_priors; ++i)
                {
                for (j = 0; j < 4; ++j)
                    {
                    top_data[idx++] = variance[j];
                    }
                }
            }
        }
}

float* preprocessed
    (
    objdetect_struct* objdet_wksp
    )
{
    int netw = objdet_wksp->net.w;
    int neth = objdet_wksp->net.h;
    unsigned char* src = objdet_wksp->src;
    const float premean = 127.5f;
    const float prediv = 0.007843f;
    float *imbuf = (float *)alloc_from_stack(netw * neth * 3 * sizeof(float));
    // split into whole b, whole g, whole r buffer, and normalize
    int idx = 0;
    int flatsize = netw * neth;
    float* imbufptr = imbuf;
    unsigned char* srcptr = src;
    // b
    for(idx = 0; idx < flatsize; ++idx)
        {
        imbufptr[idx] = ((*srcptr) - premean) * prediv;
        srcptr += 3;
        }
    // g
    imbufptr = imbuf + flatsize;
    srcptr = src + 1;
    for(idx = 0; idx < flatsize; ++idx)
        {
        imbufptr[idx] = ((*srcptr) - premean) * prediv;
        srcptr += 3;
        }
    // r
    imbufptr = imbuf + (flatsize << 1);
    srcptr = src + 2;
    for(idx = 0; idx < flatsize; ++idx)
        {
        imbufptr[idx] = ((*srcptr) - premean) * prediv;
        srcptr += 3;
        }
    return imbuf;
}

void network_predict
    (
    network_struct net
    )
{
    // forward propagation
    int i = 0;
    for(i = 0; i < net.n; ++i)
        {
        layer_struct l = net.layers[i];
        l.forward(l, net);
        net.input = l.output;
        }
}

int nms_comparator
    (
    const void *pa,
    const void *pb
    )
{
    score_index_struct a = *(score_index_struct *)pa;
    score_index_struct b = *(score_index_struct *)pb;
    float diff = a.score - b.score;
    if(diff < 0)
        {
        return 1;
        }
    else if(diff > 0)
        {
        return -1;
        }
    return 0;
}

void get_detection_out
    (
    objdetect_struct* objdet_wksp
    )
{
    network_struct net = objdet_wksp->net;
    float *mbox_conf_ptr = net.layers[84].output;
    const int num_classes = objdet_wksp->class_num + 1; // 1 for background
    const int num_conf = net.layers[84].outputs / num_classes;
    int i = 0;
    int j = 0;
    int k = 0;
    // softmax on mbox_conf
    for (i = 0; i < num_conf; ++i)
        {
        float maxconf = 0.0f;
        float sumconf = 0.0f;
        for (j = 0; j < num_classes; ++j)
            {
            if (mbox_conf_ptr[j] > maxconf)
                {
                maxconf = mbox_conf_ptr[j];
                }
            }
        for (j = 0; j < num_classes; ++j)
            {
            mbox_conf_ptr[j] = exp(mbox_conf_ptr[j] - maxconf);
            sumconf += mbox_conf_ptr[j];
            }
        for (j = 0; j < num_classes; ++j)
            {
            mbox_conf_ptr[j] /= sumconf;
            }
        mbox_conf_ptr += num_classes;
        }


    const float* loc_data = net.layers[83].output;
    const float* conf_data = net.layers[84].output;
    const float* prior_data = net.layers[85].output;
    const int num_priors = (net.layers[85].outputs >> 3);
    // get bbox candidates, note x, y here are center x, y
    box_struct *decode_bboxes = (box_struct *)alloc_from_stack(num_priors * sizeof(box_struct));
    for (i = 0; i < num_priors; ++i)
        {
        int start_idx = i * 4;
        const float prior_bbox_xmin = prior_data[start_idx];
        const float prior_bbox_ymin = prior_data[start_idx + 1];
        const float prior_bbox_xmax = prior_data[start_idx + 2];
        const float prior_bbox_ymax = prior_data[start_idx + 3];
        const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
        const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
        const float prior_center_x = (prior_bbox_xmin + prior_bbox_xmax) / 2.0f;
        const float prior_center_y = (prior_bbox_ymin + prior_bbox_ymax) / 2.0f;
        const float bbox_xmin = loc_data[start_idx];
        const float bbox_ymin = loc_data[start_idx + 1];
        const float bbox_xmax = loc_data[start_idx + 2];
        const float bbox_ymax = loc_data[start_idx + 3];
        const int shift_to_var = 4 * num_priors;
        decode_bboxes[i].x = prior_data[shift_to_var + start_idx] * bbox_xmin * prior_width + prior_center_x;
        decode_bboxes[i].y = prior_data[shift_to_var + start_idx + 1] * bbox_ymin * prior_height + prior_center_y;
        decode_bboxes[i].w = exp(prior_data[shift_to_var + start_idx + 2] * bbox_xmax) * prior_width;
        decode_bboxes[i].h = exp(prior_data[shift_to_var + start_idx + 3] * bbox_ymax) * prior_height;
        }
    // do nms
    const float score_threshold = objdet_wksp->thresh;
    const float nms_threshold = objdet_wksp->nms_thresh;
    //const float score_threshold = 0.35f;
    //const float nms_threshold = 0.75f;
    const int topk = 100;
    int num_resobj = 0;

    const int imw = objdet_wksp->srcw;
    const int imh = objdet_wksp->srch;

    int num_rest = 0;
    score_index_struct *score_index_vec = (score_index_struct *)alloc_from_stack(num_priors * sizeof(score_index_struct));
    const float *scores = conf_data + 1; // person
    for (j = 0; j < num_priors; ++j)
        {
        if (*scores > score_threshold)
            {
            score_index_vec[num_rest].index = j;
            score_index_vec[num_rest].score = *scores;
            ++num_rest;
            }
        scores += num_classes;
        }
    qsort(score_index_vec, num_rest, sizeof(score_index_struct), nms_comparator);
    int veclength = num_rest > topk ? topk : num_rest;
    //nms
    for(j = 0; j < veclength; ++j)
        {
        if(score_index_vec[j].index == -1)
            {
            continue;
            }
        box_struct a = decode_bboxes[score_index_vec[j].index];
        for(k = j + 1; k < veclength; ++k)
            {
            box_struct b = decode_bboxes[score_index_vec[k].index];
            float overlap = box_iou(a, b);
            if(overlap > nms_threshold)
                {
                score_index_vec[k].index = -1;
                }
            }
        }
    //fill in and resize back to result
    for(j = 0; j < veclength; ++j)
        {
        if(score_index_vec[j].index == -1)
            {
            continue;
            }
        const float score = score_index_vec[j].score;
        //printf("prob: %f\n", score);

        box_struct a = decode_bboxes[score_index_vec[j].index];
        const int width = a.w * imw;
        const int height = a.h * imh;
        const int xmin = a.x * imw - (width >> 1);
        const int ymin = a.y * imh - (height >> 1);
        objdet_wksp->output[5 * num_resobj + 1] = xmin;
        objdet_wksp->output[5 * num_resobj + 2] = ymin;
        objdet_wksp->output[5 * num_resobj + 3] = width;
        objdet_wksp->output[5 * num_resobj + 4] = height;
        objdet_wksp->output[5 * num_resobj + 5] = (int)(100 * score);
        ++num_resobj;
        }

    partial_free_from_stack(num_priors * sizeof(score_index_struct));

    objdet_wksp->output[0] = num_resobj;

    partial_free_from_stack(num_priors * sizeof(box_struct));
}

void clear_network
    (
    network_struct net
    )
{
    int i = 0;
    for(i = 0; i < net.n; ++i)
        {
        layer_struct l = net.layers[i];
        memset(l.output, 0, l.outputs * sizeof(float));
        }
}
