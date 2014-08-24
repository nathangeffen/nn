
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

struct test_series {
	unsigned cases;
	unsigned successes;
	unsigned failures;
	char description[256];
	int verbose;
	FILE *log;
};

struct test_series *test_init(const char *description,
			      const int verbose,
			      FILE *log)
{
	struct test_series *t = malloc(sizeof(*t));
	if (t == NULL)
		return NULL;
	t->cases = t->successes = t->failures = 0;
	strncpy(t->description, description, 255);
	t->description[255] = '\0';
	t->verbose = verbose;
	if (log == NULL)
		t->log = stderr;
	else
		t->log = log;
	return t;
}

bool
test(struct test_series *t,
     bool expr,
     const char *description,
     const char *file,
     const int line)
{
	++t->cases;
	if (expr) {
		++t->successes;
		return true;
	} else {
		++t->failures;
		if (t->verbose)
			fprintf(t->log, "FAIL:\t%s %d\n", file, line);
		return false;
	}
}

unsigned
cases(const struct test_series *t)
{
	return t->cases;
}

unsigned
successes(const struct test_series *t)
{
	return t->successes;
}

unsigned
failures(const struct test_series *t)
{
	return t->failures;
}

FILE *test_log(const struct test_series *t)
{
	return t->log;
}

void test_summary(const struct test_series *test)
{
	fprintf(test->log, "Tests: %u\tSuccesses: %u\tFailures %u\n",
		test->cases, test->successes, test->failures);
}

void test_destroy(struct test_series *test)
{
	free(test);
}
