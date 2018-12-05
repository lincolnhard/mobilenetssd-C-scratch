#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"
#include "objdetect_pub.h"

int main
    (
    void
    )
{
    init_stack();
    const int netw = 300;
    const int neth = 300;
    objdetect_init("mobilenetssd-person.weights", netw, neth);

    cv::VideoCapture cap;
    cap.open(0);
    cv::Mat im;
    while (1)
        {
        cap >> im;
        cv::Mat netim;
        cv::resize(im, netim, cv::Size(netw, neth));

        double time1 = what_time_is_it_now();

        int *out = objdetect_main(netim.data, im.cols, im.rows);

        double time2 = what_time_is_it_now();
        std::cout << time2 - time1 << std::endl;
#if 0
        int i = 0;
        for (i = 0; i < out[0]; ++i)
            {
            const int x = out[i*5+1];
            const int y = out[i*5+2];
            const int w = out[i*5+3];
            const int h = out[i*5+4];
            const int p = out[i*5+5];
            cv::rectangle(im, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0));
            }
        cv::imshow("demo", im);
        unsigned char key = cv::waitKey(1);
        if (key == 27)
            {
            break;
            }
#endif
        }

    objdetect_free();
    free_stack();
    return 0;
}
