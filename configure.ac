## configure.ac

AC_INIT([libnn], [0.1], [nathangeffen@quackdown.info], [nn])
AC_PREREQ([2.69])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([1.10 subdir-objects -Wall no-define])
AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_HEADERS([config.h])
AM_PROG_AR
AC_PROG_CC
PKG_CHECK_MODULES([GSL], [gsl])

LT_INIT([disable-static])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_FUNC_MALLOC
AC_CHECK_FUNCS([floor])
AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([pow])
AC_CHECK_FUNCS([strchr])
AC_CHECK_HEADERS([float.h])
AC_CHECK_HEADERS([limits.h])
AC_CHECK_HEADERS([stddef.h])
AC_CHECK_TYPES([ptrdiff_t])
AC_TYPE_SIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T

# Define these substitions here to keep all version information in one place.
# For information on how to properly maintain the library version information,
# refer to the libtool manual, section "Updating library version information":
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
AC_SUBST([NN_SO_VERSION], [1:1:0])
AC_SUBST([NN_API_VERSION], [0.1])
AC_CONFIG_FILES([Makefile
                 nn-${NN_API_VERSION}.pc:nn.pc.in])
AC_OUTPUT
