#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define BLOCK_BYTES (1UL << 20)
#define AIO_DEPTH 128

struct run {
	const char *path;
	unsigned char *dst;
	size_t bytes;
	int direct;
	atomic_size_t next_block;
	atomic_int error;
};

struct worker {
	struct run *run;
};

static long io_setup(unsigned int nr, aio_context_t *ctx)
{
	return syscall(SYS_io_setup, nr, ctx);
}

static long io_destroy(aio_context_t ctx)
{
	return syscall(SYS_io_destroy, ctx);
}

static long io_submit(aio_context_t ctx, long nr, struct iocb **iocbs)
{
	return syscall(SYS_io_submit, ctx, nr, iocbs);
}

static long io_getevents(aio_context_t ctx, long min_nr, long nr,
			 struct io_event *events)
{
	return syscall(SYS_io_getevents, ctx, min_nr, nr, events, NULL);
}

static int prepare(struct run *run, int fd, struct iocb *cb, size_t slot)
{
	size_t block = atomic_fetch_add_explicit(&run->next_block, 1,
						memory_order_relaxed);
	size_t offset = block * BLOCK_BYTES;
	size_t length;

	if (offset >= run->bytes)
		return 0;
	length = run->bytes - offset;
	if (length > BLOCK_BYTES)
		length = BLOCK_BYTES;
	memset(cb, 0, sizeof(*cb));
	cb->aio_data = slot;
	cb->aio_lio_opcode = IOCB_CMD_PREAD;
	cb->aio_fildes = fd;
	cb->aio_buf = (uint64_t)(run->dst + offset);
	cb->aio_nbytes = length;
	cb->aio_offset = offset;
	return 1;
}

static int submit_all(aio_context_t ctx, struct iocb **pending, size_t count)
{
	size_t done = 0;

	while (done < count) {
		long n = io_submit(ctx, count - done, pending + done);

		if (n <= 0)
			return -1;
		done += n;
	}
	return 0;
}

static void *read_worker(void *opaque)
{
	struct run *run = ((struct worker *)opaque)->run;
	struct iocb cbs[AIO_DEPTH];
	struct iocb *pending[AIO_DEPTH];
	struct io_event events[AIO_DEPTH];
	size_t active = 0, count = 0, i;
	aio_context_t ctx = 0;
	int fd = -1;

	fd = open(run->path, O_RDONLY | O_CLOEXEC | (run->direct ? O_DIRECT : 0));
	if (fd < 0 || io_setup(AIO_DEPTH, &ctx) < 0)
		goto fail;
	for (i = 0; i < AIO_DEPTH; i++) {
		if (!prepare(run, fd, &cbs[i], i))
			break;
		pending[count++] = &cbs[i];
	}
	if (submit_all(ctx, pending, count))
		goto fail;
	active = count;
	while (active) {
		long n = io_getevents(ctx, 1, AIO_DEPTH, events);

		if (n <= 0)
			goto fail;
		count = 0;
		for (i = 0; i < (size_t)n; i++) {
			size_t slot = events[i].data;

			if (slot >= AIO_DEPTH || events[i].res < 0 ||
			    (uint64_t)events[i].res != cbs[slot].aio_nbytes)
				goto fail;
			active--;
			if (prepare(run, fd, &cbs[slot], slot)) {
				pending[count++] = &cbs[slot];
				active++;
			}
		}
		if (submit_all(ctx, pending, count))
			goto fail;
	}
	io_destroy(ctx);
	close(fd);
	return NULL;

fail:
	atomic_store(&run->error, errno ? errno : EIO);
	if (ctx)
		io_destroy(ctx);
	if (fd >= 0)
		close(fd);
	return NULL;
}

static double elapsed(const struct timespec *start, const struct timespec *end)
{
	return end->tv_sec - start->tv_sec +
	       (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(int argc, char **argv)
{
	struct run run = {.path = argc > 1 ? argv[1] : NULL};
	struct stat st;
	struct timespec start, end;
	struct rusage before, after;
	struct worker *workers;
	pthread_t *threads;
	unsigned long checksum = 0;
	long thread_count;
	size_t i;
	double seconds;

	if (argc != 4) {
		fprintf(stderr, "usage: %s FILE THREADS direct|buffered\n", argv[0]);
		return 64;
	}
	if (!strcmp(argv[3], "direct"))
		run.direct = 1;
	else if (strcmp(argv[3], "buffered"))
		return 64;
	thread_count = strtol(argv[2], NULL, 10);
	if (thread_count < 1 || thread_count > 32 || stat(run.path, &st) ||
	    st.st_size < (off_t)BLOCK_BYTES) {
		perror("invalid input");
		return 64;
	}
	run.bytes = (size_t)st.st_size & ~(BLOCK_BYTES - 1);
	run.dst = mmap(NULL, run.bytes, PROT_READ | PROT_WRITE,
		       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (run.dst == MAP_FAILED) {
		perror("mmap");
		return 1;
	}
	madvise(run.dst, run.bytes, MADV_NOHUGEPAGE);
	workers = calloc(thread_count, sizeof(*workers));
	threads = calloc(thread_count, sizeof(*threads));
	if (!workers || !threads)
		return 1;
	getrusage(RUSAGE_SELF, &before);
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < (size_t)thread_count; i++) {
		workers[i].run = &run;
		if (pthread_create(&threads[i], NULL, read_worker, &workers[i]))
			return 1;
	}
	for (i = 0; i < (size_t)thread_count; i++)
		pthread_join(threads[i], NULL);
	clock_gettime(CLOCK_MONOTONIC, &end);
	getrusage(RUSAGE_SELF, &after);
	if (atomic_load(&run.error)) {
		errno = atomic_load(&run.error);
		perror("native AIO read");
		return 1;
	}
	for (i = 0; i < run.bytes; i += BLOCK_BYTES)
		checksum += run.dst[i];
	seconds = elapsed(&start, &end);
	printf("{\"threads\":%ld,\"bytes\":%zu,\"seconds\":%.6f,"
	       "\"gib_per_s\":%.3f,\"minor_faults\":%ld,"
	       "\"major_faults\":%ld,\"checksum\":%lu}\n",
	       thread_count, run.bytes, seconds,
	       run.bytes / seconds / (1024.0 * 1024.0 * 1024.0),
	       after.ru_minflt - before.ru_minflt,
	       after.ru_majflt - before.ru_majflt, checksum);
	munmap(run.dst, run.bytes);
	free(threads);
	free(workers);
	return 0;
}
