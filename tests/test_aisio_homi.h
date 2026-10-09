/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_aisio_homi.h - Shared setup for aisio tests against a live HOMI stack.
 *
 * The aisio backend no longer owns the controller or takes injected
 * extents: it joins the multi-process group of a running homi server and
 * resolves a file's extents from the index xal-server publishes over shared
 * memory. These tests therefore run while the homi/qublk/xal-server stack
 * is up and the test filesystem is mounted (over /dev/ublkbN). The harness
 * (tasks/test.yaml) brings the stack up, exports OPENDS_HOMI_DEV /
 * OPENDS_XAL_SHM / OPENDS_HOMI_MNT, and passes the path of a real file on
 * the mount.
 *
 * Each test opens that file, registers it (which resolves its extents from
 * the shared index), and runs the backend-agnostic test logic.
 */
#ifndef TEST_AISIO_HOMI_H_
#define TEST_AISIO_HOMI_H_

#include "opends.h"

#include <cuda.h>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

struct aisio_homi {
	CUcontext cuctx;
	int fd;
	opends_handle_t fh;
};

/*
 * Create a CUDA context (the upcie-cuda backend needs one current for
 * device init and buffer allocation), open the aisio driver against the
 * HOMI stack named by the environment, open `path`, and register it.
 *
 * Returns 0 on success with *out filled in, or -1 on failure (a message is
 * printed). On failure nothing needs tearing down.
 *
 * `flags` is passed to open(2) and must include O_DIRECT; read tests use
 * O_RDONLY, write tests O_RDWR (optionally O_CREAT|O_TRUNC). Extents are
 * resolved per I/O (not at registration), so a freshly created file just
 * works: the re-index the first opends_sync_write performs makes its blocks
 * resolvable.
 */
static inline int
aisio_homi_setup_flags(const char *path, int flags, struct aisio_homi *out)
{
	CUdevice cudev;

	out->cuctx = NULL;
	out->fd = -1;
	out->fh = NULL;

	cuInit(0);
	cuDeviceGet(&cudev, 0);
	if (cuCtxCreate(&out->cuctx, 0, cudev) != CUDA_SUCCESS) {
		fprintf(stderr, "cuCtxCreate failed\n");
		return -1;
	}

	opends_error_t err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr,
		        "driver_open: %s (is the homi stack running and "
		        "OPENDS_HOMI_DEV set?)\n",
		        opends_op_status_error(err.err));
		cuCtxDestroy(out->cuctx);
		return -1;
	}

	out->fd = open(path, flags, 0644);
	if (out->fd < 0) {
		fprintf(stderr, "open(%s) failed\n", path);
		opends_driver_close();
		cuCtxDestroy(out->cuctx);
		return -1;
	}

	err = opends_handle_register(&out->fh, out->fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "handle_register: %s\n",
		        opends_op_status_error(err.err));
		close(out->fd);
		opends_driver_close();
		cuCtxDestroy(out->cuctx);
		return -1;
	}

	return 0;
}

static inline int
aisio_homi_setup(const char *path, struct aisio_homi *out)
{
	return aisio_homi_setup_flags(path, O_RDONLY | O_DIRECT, out);
}

static inline void
aisio_homi_teardown(struct aisio_homi *a)
{
	if (a->fh)
		opends_handle_deregister(a->fh);
	opends_driver_close();
	if (a->fd >= 0)
		close(a->fd);
	if (a->cuctx)
		cuCtxDestroy(a->cuctx);
}

#endif /* TEST_AISIO_HOMI_H_ */
