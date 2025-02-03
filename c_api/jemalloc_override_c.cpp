#ifdef FAISS_USE_JEMALLOC

#define JEMALLOC_MANGLE
#include <jemalloc/jemalloc.h>

#endif /* FAISS_USE_JEMALLOC */
