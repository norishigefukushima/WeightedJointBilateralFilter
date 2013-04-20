#ifndef _CONFIG_H____
#define _CONFIG_H____

//for TBB parallelization. If you do not have TBB, please comment out.
#define HAVE_TBB

//for SSE4.1 SIMD optimization. If you do not have SSE4.1 capable CPU, please comment out.
#define CV_SSE4_1 1

#if CV_SSE4_1
#include <smmintrin.h>
#endif

#endif