#!/bin/bash

set -ex

CLANG_VERSION_MAJOR=10

APT_DEPENDENCIES=(
	build-essential
	clang-${CLANG_VERSION_MAJOR}
	ninja-build
	python3
	python3-dev
	python3-numpy
	python3-typing-extensions
	python3-pip
)

function elevated_exec() {
	sudo "$@"
}

elevated_exec apt update
elevated_exec apt upgrade -y

elevated_exec apt install -y "${APT_DEPENDENCIES[@]}"

elevated_exec apt remove cmake -y
elevated_exec snap install cmake --classic

elevated_exec apt autoremove -y

function update_alternatives() {
	local from=$1
	local to=$2
	local name="$(basename "$from")"
	elevated_exec update-alternatives --remove-all "$name" || true
	elevated_exec update-alternatives --install "$from" "$name" "$to" 100
}

update_alternatives /usr/bin/python /usr/bin/python3
update_alternatives /usr/bin/pip /usr/bin/pip3
update_alternatives /usr/bin/cc "/usr/bin/clang-${CLANG_VERSION_MAJOR}"
update_alternatives /usr/bin/c++ "/usr/bin/clang++-${CLANG_VERSION_MAJOR}"
update_alternatives /usr/bin/clang "/usr/bin/clang-${CLANG_VERSION_MAJOR}"
update_alternatives /usr/bin/clang++ "/usr/bin/clang++-${CLANG_VERSION_MAJOR}"

set +x

for prefix in "" /usr/bin/; do
	for bin_name in cc c++ clang clang++; do
		bin_path="${prefix}${bin_name}"
		actual_version="$(echo '__clang_major__' | "${bin_path}" -E - | tail -n1 || echo 'preprocessor error')"
		if [[ "$actual_version" != "$CLANG_VERSION_MAJOR" ]]; then
			echo "${bin_path} reported version '${actual_version}' but '${CLANG_VERSION_MAJOR}' is expected"
		fi
	done
done

set -x

pip install torchvision