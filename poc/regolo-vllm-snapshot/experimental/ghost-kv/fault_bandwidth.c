#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

struct worker {
	char *start;
	size_t bytes;
};

static void *fault_pages(void *arg)
{
	struct worker *worker = arg;
	size_t page = (size_t)sysconf(_SC_PAGESIZE);

	for (size_t offset = 0; offset < worker->bytes; offset += page)
		worker->start[offset] = 1;
	return NULL;
}

static double seconds(struct timespec start, struct timespec end)
{
	return end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		fprintf(stderr, "usage: %s BYTES THREADS\n", argv[0]);
		return 2;
	}
	size_t bytes = strtoull(argv[1], NULL, 10);
	unsigned int threads = strtoul(argv[2], NULL, 10);
	size_t page = (size_t)sysconf(_SC_PAGESIZE);
	if (!bytes || !threads || bytes % page) {
		fprintf(stderr, "bytes must be nonzero and page-aligned; threads must be nonzero\n");
		return 2;
	}
	char *memory = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
			    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (memory == MAP_FAILED) {
		perror("mmap");
		return 1;
	}
	if (madvise(memory, bytes, MADV_NOHUGEPAGE)) {
		perror("madvise");
		return 1;
	}
	pthread_t *ids = calloc(threads, sizeof(*ids));
	struct worker *workers = calloc(threads, sizeof(*workers));
	if (!ids || !workers)
		return 1;
	size_t pages = bytes / page;
	struct rusage before, after;
	struct timespec start, end;
	getrusage(RUSAGE_SELF, &before);
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (unsigned int i = 0; i < threads; i++) {
		size_t first = pages * i / threads;
		size_t last = pages * (i + 1) / threads;
		workers[i] = (struct worker){memory + first * page, (last - first) * page};
		int rc = pthread_create(&ids[i], NULL, fault_pages, &workers[i]);
		if (rc) {
			errno = rc;
			perror("pthread_create");
			return 1;
		}
	}
	for (unsigned int i = 0; i < threads; i++)
		pthread_join(ids[i], NULL);
	clock_gettime(CLOCK_MONOTONIC, &end);
	getrusage(RUSAGE_SELF, &after);
	double elapsed = seconds(start, end);
	printf("{\"bytes\":%zu,\"threads\":%u,\"seconds\":%.6f,"
	       "\"gib_per_s\":%.6f,\"minor_faults\":%ld,\"major_faults\":%ld}\n",
	       bytes, threads, elapsed, bytes / elapsed / (1ULL << 30),
	       after.ru_minflt - before.ru_minflt, after.ru_majflt - before.ru_majflt);
	munmap(memory, bytes);
	free(workers);
	free(ids);
	return 0;
}
