//  Copyright 2024-Present Couchbase, Inc.
//
//  Use of this software is governed by the Business Source License included
//  in the file licenses/BSL-Couchbase.txt.  As of the Change Date specified
//  in that file, in accordance with the Business Source License, use of this
//  software will be governed by the Apache License, Version 2.0, included in
//  the file licenses/APL2.txt.

#include "jemalloc_stat.h"

// Helper function for calling jemalloc epoch.
// jemalloc stats are not updated until the caller requests a synchronisation,
// which is done by the epoch call.
static void callJemallocEpoch() {
    size_t epoch = 1;
    size_t sz = sizeof(epoch);
    // The return of epoch is the current epoch, which we don't make use of
    mallctl("epoch", &epoch, &sz, &epoch, sz);
}

size_t jm_allocated() {
    // call mallctl defined in jemalloc
    // first refresh the value of the stats
    callJemallocEpoch();
    // now get the stats
    size_t allocated, sz;
    sz = sizeof(size_t);
    mallctl("stats.allocated", &allocated, &sz, NULL, 0);
    return allocated;
}

size_t jm_resident() {
    // call mallctl defined in jemalloc
    // first refresh the value of the stats
    callJemallocEpoch();
    // now get the stats
    size_t resident, sz;
    sz = sizeof(size_t);
    mallctl("stats.resident", &resident, &sz, NULL, 0);
    return resident;
}