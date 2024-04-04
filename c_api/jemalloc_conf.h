//  Copyright 2024-Present Couchbase, Inc.
//
//  Use of this software is governed by the Business Source License included
//  in the file licenses/BSL-Couchbase.txt.  As of the Change Date specified
//  in that file, in accordance with the Business Source License, use of this
//  software will be governed by the Apache License, Version 2.0, included in
//  the file licenses/APL2.txt.

#ifndef JEMALLOC_CONF_H
#define JEMALLOC_CONF_H

#include <jemalloc/jemalloc_faiss.h>
/* jemalloc checks for this symbol, and it's contents for the config to use. */
const char* malloc_conf =
#ifndef __APPLE__
        /* Enable background worker thread for asynchronous purging.
        * Background threads are non-functional in jemalloc 5.1.0 on macOS due to
        * implementation discrepancies between the background threads and mutexes.
        * https://github.com/jemalloc/jemalloc/issues/1433
        */
        "background_thread:true,"
#endif
        /* Use 4 arenas, instead of the default based on number of CPUs.
        Helps to minimize heap fragmentation.
        https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md
        */
        "narenas:4,"
#ifdef __linux__
        /* Start with profiling enabled but inactive; this allows us to
        turn it on/off at runtime. */
        "prof:true,prof_active:false,"
#endif
        /* abort immediately on illegal options, just for sanity */
        "abort_conf:true";

#endif /* JEMALLOC_CONF_H */
