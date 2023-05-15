#!/bin/bash
# Called from build.sh

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

# use all flags used by ubuntu 20.04 for hardening builds, dpkg-buildflags --export
# other flags mentioned in https://wiki.ubuntu.com/ToolChain/CompilerFlags can't be
# used because the distros used here are too old
MANYLINUX_CPPFLAGS="-Wdate-time -D_FORTIFY_SOURCE=2"
MANYLINUX_CFLAGS="-g -O2 -Wall -fdebug-prefix-map=/=. -fstack-protector-strong -Wformat -Werror=format-security"
MANYLINUX_CXXFLAGS="-g -O2 -Wall -fdebug-prefix-map=/=. -fstack-protector-strong -Wformat -Werror=format-security"
MANYLINUX_LDFLAGS="-Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,now"

# Install a more recent openssl
check_var ${OPENSSL_ROOT}
check_var ${OPENSSL_HASH}
check_var ${OPENSSL_DOWNLOAD_URL}

OPENSSL_VERSION=${OPENSSL_ROOT#*-}
OPENSSL_MIN_VERSION=1.1.1

# || test $? -eq 141 is there to ignore SIGPIPE with set -o pipefail
# c.f. https://stackoverflow.com/questions/22464786/ignoring-bash-pipefail-for-error-code-141#comment60412687_33026977
INSTALLED=$((openssl version | head -1 || test $? -eq 141) | awk '{ print $2 }')
SMALLEST=$(echo -e "${INSTALLED}\n${OPENSSL_MIN_VERSION}" | sort -t. -k 1,1n -k 2,2n -k 3,3n -k 4,4n | head -1 || test $? -eq 141)

# Ignore letters in version numbers
if [ "${SMALLEST}" = "${OPENSSL_MIN_VERSION}" ]; then
	echo "skipping installation of openssl ${OPENSSL_VERSION}, system provides openssl ${INSTALLED} which is newer than openssl ${OPENSSL_MIN_VERSION}"
	exit 0
fi

yum erase -y openssl-devel

fetch_source ${OPENSSL_ROOT}.tar.gz ${OPENSSL_DOWNLOAD_URL}
check_sha256sum ${OPENSSL_ROOT}.tar.gz ${OPENSSL_HASH}
tar -xzf ${OPENSSL_ROOT}.tar.gz
pushd ${OPENSSL_ROOT}
./config no-shared --prefix=/usr/local/ssl --openssldir=/usr/local/ssl CPPFLAGS="${MANYLINUX_CPPFLAGS}" CFLAGS="${MANYLINUX_CFLAGS} -fPIC" CXXFLAGS="${MANYLINUX_CXXFLAGS} -fPIC" LDFLAGS="${MANYLINUX_LDFLAGS} -fPIC" > /dev/null
make > /dev/null
make install_sw > /dev/null
popd
rm -rf ${OPENSSL_ROOT} ${OPENSSL_ROOT}.tar.gz


/usr/local/ssl/bin/openssl version