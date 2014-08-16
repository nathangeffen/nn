## Makefile.am for libnn
## Based on LibExample from
## http://www.openismus.com/documents/linux/building_libraries/building_libraries

## Place generated object files (.o) into the same directory as their source
## files, in order to avoid collisions when non-recursive make is used.

AUTOMAKE_OPTIONS = subdir-objects
AM_CFLAGS = -std=c99

ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libnn-@NN_API_VERSION@.la
libnn_@NN_API_VERSION@_la_SOURCES = nn/nn.c nn/nn.h nn/io.c nn/cJSON/cJSON.c
	nn/cJSON.h uthash/uthash.h
libnn_@NN_API_VERSION@_la_CFLAGS = $(GSL_CFLAGS) -std=c99
libnn_@NN_API_VERSION@_la_LDFLAGS = -version-info $(NN_SO_VERSION) $(GSL_LIBS)

bin_PROGRAMS = testnn
testnn_SOURCES = src/testnn.c src/test.h src/test.c
testnn_LDADD = libnn-@NN_API_VERSION@.la
testnn_CFLAGS = $(GSL_CFLAGS) -std=c99
testnn_LDFLAGS = $(GSL_LIBS)

nn_includedir = $(includedir)/nn-$(NN_API_VERSION)

nobase_nn_include_HEADERS = nn/nn.h nn/cJSON/cJSON.h nn/uthash/uthash.h

dist_doc_data = README
