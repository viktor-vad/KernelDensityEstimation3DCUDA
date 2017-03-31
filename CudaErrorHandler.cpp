#include "CudaErrorHandler.h"
//
void default_kde_cuda_error_handler(char const *const func, const char *const file, int const line,const char* message)
{
    fprintf(stderr, "CUDA error at %s:%d in func: %s msg: %s \n", file, line, func, message);
    exit(EXIT_FAILURE);
}

std::function<void(char const *const func, const char *const file, int const line, const char* message)> CudaErrorHandler::KDE_CUDA_ERROR_HANDLER_FUNC = default_kde_cuda_error_handler;
