# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import setuptools
import os
import sys
import datetime

nightly_build = False
package_name = 'ort_moe'

if '--nightly_build' in sys.argv:
    package_name = 'ort_moe-nightly'
    nightly_build = True
    sys.argv.remove('--nightly_build')

version_number = ''
with open('VERSION_NUMBER') as f:
    version_number = f.readline().strip()

if nightly_build:
    #https://docs.microsoft.com/en-us/azure/devops/pipelines/build/variables
    build_suffix = os.environ.get('BUILD_BUILDNUMBER')
    if build_suffix is None:
      #The following line is only for local testing
      build_suffix = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
    else:
      build_suffix = build_suffix.replace('.','')
    version_number = version_number + ".dev" + build_suffix
else:
    build_suffix = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
    version_number = version_number + ".dev" + build_suffix


if __name__ == '__main__':
    setuptools.setup(
        name=package_name,
        version=version_number,
        packages=['ort_moe'],
        )

