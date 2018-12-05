#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"
#include "objdetect_pub.h"


void trim(char *str)
{
    char buffer[4096] = {0};
    sprintf(buffer, "%s", str);

    char *p = buffer;
    while (*p == ' ' || *p == '\t') ++p;

    char *end = p + strlen(p) - 1;
    while (*end == ' ' || *end == '\t') {
        *end = '\0';
        --end;
    }
    sprintf(str, "%s", p);
}

void find_replace(char *str, char *orig, char *rep, char *output)
{
    char buffer[4096] = {0};
    char *p;

    sprintf(buffer, "%s", str);
    if(!(p = strstr(buffer, orig))){  // Is 'orig' even in 'str'?
        sprintf(output, "%s", str);
        return;
    }

    *p = '\0';

    sprintf(output, "%s%s%s", buffer, rep, p+strlen(orig));
}

void find_replace_extension(char *str, char *orig, char *rep, char *output)
{
    char *buffer = (char *)calloc(4096, sizeof(char));

    sprintf(buffer, "%s", str);
    char *p = strstr(buffer, orig);
    int offset = (p - buffer);
    int chars_from_end = strlen(buffer) - offset;
    if (!p || chars_from_end != strlen(orig)) {  // Is 'orig' even in 'str' AND is 'orig' found at the end of 'str'?
        sprintf(output, "%s", str);
        free(buffer);
        return;
    }

    *p = '\0';
    sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
    free(buffer);
}

void replace_image_to_label(char *input_path, char *output_path)
{
    find_replace(input_path, "/images/train2014/", "/labels/train2014/", output_path);    // COCO
    find_replace(output_path, "/images/val2014/", "/labels/val2014/", output_path);        // COCO
    find_replace(output_path, "/JPEGImages/", "/labels/", output_path);    // PascalVOC
    find_replace(output_path, "\\images\\train2014\\", "\\labels\\train2014\\", output_path);    // COCO
    find_replace(output_path, "\\images\\val2014\\", "\\labels\\val2014\\", output_path);        // COCO
    find_replace(output_path, "\\JPEGImages\\", "\\labels\\", output_path);    // PascalVOC
    //find_replace(output_path, "/images/", "/labels/", output_path);    // COCO
    //find_replace(output_path, "/VOC2007/JPEGImages/", "/VOC2007/labels/", output_path);        // PascalVOC
    //find_replace(output_path, "/VOC2012/JPEGImages/", "/VOC2012/labels/", output_path);        // PascalVOC

    //find_replace(output_path, "/raw/", "/labels/", output_path);
    trim(output_path);

    // replace only ext of files
    find_replace_extension(output_path, ".jpg", ".txt", output_path);
    find_replace_extension(output_path, ".JPG", ".txt", output_path); // error
    find_replace_extension(output_path, ".jpeg", ".txt", output_path);
    find_replace_extension(output_path, ".JPEG", ".txt", output_path);
    find_replace_extension(output_path, ".png", ".txt", output_path);
    find_replace_extension(output_path, ".PNG", ".txt", output_path);
    find_replace_extension(output_path, ".bmp", ".txt", output_path);
    find_replace_extension(output_path, ".BMP", ".txt", output_path);
    find_replace_extension(output_path, ".ppm", ".txt", output_path);
    find_replace_extension(output_path, ".PPM", ".txt", output_path);
}

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

box_label *read_boxes(char *filename, int *n)
{
    FILE *file = fopen(filename, "r");
    if(!file)
        {
        fprintf(stderr, "Couldn't open file: %s\n", filename);
        exit(0);
        }
    float x, y, h, w;
    int id;
    int count = 0;
    int size = 64;
    box_label *boxes = (box_label *)calloc(size, sizeof(box_label));
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        if(count == size) {
            size = size * 2;
            boxes = (box_label *)realloc(boxes, size*sizeof(box_label));
        }
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

typedef struct{
    float x, y, w, h;
} box;

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(box_prob *)pa;
    box_prob b = *(box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int main
    (
    void
    )
{
    const float iou_thresh = 0.5f;
    const float thresh_calc_avg_iou = 0.25f;
    int netw = 300;
    int neth = 300;
    init_stack();
    objdetect_init("/home/lincolnhard/Desktop/mobilenetssd-person.weights", netw, neth);

    char imgpath[128];
    char labelpath[128];
    int num_pics = 0;
    int classes = 1;
    int *truth_classes_count = (int *)calloc(classes, sizeof(int));
    FILE *fp = fopen("/media/lincolnhard/LHDISK1T/voc2/2007_test.txt", "r");
    box_prob *detections = (box_prob *)calloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    while(fgets(imgpath, 120, fp) != NULL)
        {
        size_t imgpathlen = strlen(imgpath) - 1;
        if (*imgpath && imgpath[imgpathlen] == '\n')
            {
            imgpath[imgpathlen] = '\0';
            }
        std::cout << imgpath << std::endl;
        cv::Mat im = cv::imread(imgpath);
        cv::Mat netim;
        cv::resize(im, netim, cv::Size(netw, neth));
        int *out = objdetect_main(netim.data, im.cols, im.rows);
        int nboxes = out[0];

        replace_image_to_label(imgpath, labelpath);
        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        int j = 0;
        for (j = 0; j < num_labels; ++j) {
            truth_classes_count[truth[j].id]++;
        }

        const int checkpoint_detections_count = detections_count;
        int i = 0;
        for (i = 0; i < nboxes; ++i)
            {
            int class_id = 0;
            for (class_id = 0; class_id < classes; ++class_id)
                {
                float prob = out[i*5+5];
                if (prob > 0)
                    {
                    detections_count++;
                    detections = (box_prob *)realloc(detections, detections_count * sizeof(box_prob));
                    detections[detections_count - 1].b.w = (float)out[i*5+3]/(float)im.cols;
                    detections[detections_count - 1].b.h = (float)out[i*5+4]/(float)im.rows;
                    detections[detections_count - 1].b.x = (float)out[i*5+1]/(float)im.cols + detections[detections_count - 1].b.w / 2.0f;
                    detections[detections_count - 1].b.y = (float)out[i*5+2]/(float)im.rows + detections[detections_count - 1].b.h / 2.0f;
                    detections[detections_count - 1].p = prob;
                    detections[detections_count - 1].image_index = num_pics;
                    detections[detections_count - 1].class_id = class_id;
                    detections[detections_count - 1].truth_flag = 0;
                    detections[detections_count - 1].unique_truth_index = -1;
                    int truth_index = -1;
                    float max_iou = 0;
                    for (j = 0; j < num_labels; ++j)
                        {
                        box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                        float current_iou = box_iou(detections[detections_count - 1].b, t);
                        if (current_iou > iou_thresh && class_id == truth[j].id)
                            {
                            if (current_iou > max_iou)
                                {
                                max_iou = current_iou;
                                truth_index = unique_truth_count + j;
                                }
                            }
                        }

                    // best IoU
                    if (truth_index > -1) {
                        detections[detections_count - 1].truth_flag = 1;
                        detections[detections_count - 1].unique_truth_index = truth_index;
                    }
                    // calc avg IoU, true-positives, false-positives for required Threshold
                    if (prob > thresh_calc_avg_iou)
                        {
                        int z, found = 0;
                        for (z = checkpoint_detections_count; z < detections_count-1; ++z)
                            {
                            if (detections[z].unique_truth_index == truth_index)
                                {
                                found = 1; break;
                                }
                            }
                        if(truth_index > -1 && found == 0)
                            {
                            avg_iou += max_iou;
                            ++tp_for_thresh;
                            }
                        else
                            {
                            fp_for_thresh++;
                            }
                        }
                    }
                }
            }
        unique_truth_count += num_labels;
        ++num_pics;
        }
    fclose(fp);
    std::cout << num_pics << std::endl;


    if((tp_for_thresh + fp_for_thresh) > 0)
    {
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);
    }
    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    int i = 0;
    // for PR-curve
    pr_t **pr = (pr_t **)calloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t *)calloc(detections_count, sizeof(pr_t));
    }
    printf("detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);

    int *truth_flags = (int *)calloc(unique_truth_count, sizeof(int));
    int rank;
    for (rank = 0; rank < detections_count; ++rank)
    {
        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            }
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;
        }
    }

    free(truth_flags);

    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;
        int point;
        for (point = 0; point < 11; ++point) {
            double cur_recall = point * 0.1;
            double cur_precision = 0;
            for (rank = 0; rank < detections_count; ++rank)
            {
                if (pr[i][rank].recall >= cur_recall) {    // > or >=
                    if (pr[i][rank].precision > cur_precision) {
                        cur_precision = pr[i][rank].precision;
                    }
                }
            }
            //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

            avg_precision += cur_precision;
        }
        avg_precision = avg_precision / 11;
        printf("name = person, ap = %2.2f %%\n", avg_precision*100);
        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("for thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf("for thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("mean average precision (mAP) = %f, or %2.2f %% \n", mean_average_precision, mean_average_precision * 100);

    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);

    free_stack();
    return 0;
}
