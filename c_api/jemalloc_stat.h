//  Copyright 2024-Present Couchbase, Inc.
//
//  Use of this software is governed by the Business Source License included
//  in the file licenses/BSL-Couchbase.txt.  As of the Change Date specified
//  in that file, in accordance with the Business Source License, use of this
//  software will be governed by the Apache License, Version 2.0, included in
//  the file licenses/APL2.txt.

#ifndef JEMALLOC_STAT_H
#define JEMALLOC_STAT_H

#include <stddef.h>
#include <jemalloc/jemalloc_faiss.h>
#ifdef __cplusplus
extern "C" {
#endif
    size_t jm_allocated();
    size_t jm_resident();
#ifdef __cplusplus
}
#endif


#endif /* JEMALLOC_STAT_H */
