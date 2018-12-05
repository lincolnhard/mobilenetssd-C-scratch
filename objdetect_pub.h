#ifndef ENTRYFUNC_H
#define ENTRYFUNC_H

#ifdef __cplusplus
extern "C"
    {
#endif

void objdetect_init
    (
    char* weight_file_path,
    const int netw,
    const int neth
    );

int *objdetect_main
    (
    unsigned char *im,
    int imw,
    int imh
    );

void objdetect_free
    (
    void
    );

void set_objdetect_parameter
    (
    const float in_th
    );

#ifdef __cplusplus
    }
#endif

#endif // ENTRYFUNC_H
