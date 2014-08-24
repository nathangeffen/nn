#ifndef CAG_TEST_H
#define CAG_TEST_H

#include <stdio.h>
#include <stdbool.h>

struct test_series;

struct test_series *test_init(const char *description,
			      const int verbose,
			      FILE *log);

bool test(struct test_series *t,
	  bool expr,
	  const char *description,
	  const char *file,
	  const int line);

unsigned cases(const struct test_series *t);

unsigned successes(const struct test_series *t);

unsigned failures(const struct test_series *t);

FILE *test_log(const struct test_series *t);

void test_summary(const struct test_series *test);

void test_destroy(struct test_series *test);

#define TEST(test_series, expr, desc) \
	test(test_series, expr, desc, __FILE__, __LINE__)

#define TESTCMP(test_series, ex1, cmp, ex2, desc, spec)		\
	if (TEST(test_series, ex1 cmp ex2, desc) == false) {	 \
		fprintf(test_log(test_series), "%s\t" #cmp "\t%s\n", #ex1, #ex2); \
		fprintf(test_log(test_series), spec "\t" #cmp "\t" spec, ex1, ex2); \
	}

#define TESTEQ(test_series, ex1, ex2, desc, spec)	\
	TESTCMP(test_series, ex1, ==, ex2, desc, spec)


#define TESTLT(test_series, ex1, ex2, desc, spec)	\
	TESTCMP(test_series, ex1, <, ex2, desc, spec)

#endif
