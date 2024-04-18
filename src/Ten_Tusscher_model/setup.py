#!/usr/bin/env python

# This setup.py file is largely inspired by/copied from NumPy.
"""TenTuschersinglecell!

"""

from __future__ import division, absolute_import, print_function

import sys
import os
import subprocess

from setuptools import setup, find_packages

if sys.version_info[:2] < (2, 6) or (3, 0) <= sys.version_info[0:2] < (3, 2):
    raise RuntimeError("Python version 2.6, 2.7 or >= 3.2 required.")

# General information:
DOCLINES = __doc__.split("\n")

# License would need to be fixed for PIP
CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: None
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

NAME                = 'bondarenkoclass'
MAINTAINER          = ""
MAINTAINER_EMAIL    = ""
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://bmp.ds.mpg.de"
DOWNLOAD_URL        = ""
LICENSE             = 'Copyrighted' # None yet!
CLASSIFIERS         = [_f for _f in CLASSIFIERS.split('\n') if _f]
AUTHOR              = ""
AUTHOR_EMAIL        = ""
PLATFORMS           = ["Windows", "Linux", "Mac OS-X"] # hopefully
MAJOR               = 0
MINOR               = 1
MICRO               = 1
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Install requirements, note that a plugin could require extra requirements
requires = [
]

# Optional dependencies
extras_require = {
}

# Information about what needs to be installed.
# We could force plugins to have their own setup.py (or something like that)
# I believe.
# packages can be extended manually, if that should ever be necessary.
packages = find_packages()

try:
    from setuptools.extension import Extension
    from Cython.Distutils import build_ext

    # This might be problematic with dependencies if numpy is not installed
    # yet?
    import numpy as np

    ext_packages = [
        Extension("TenTuschersinglecell.model2012",
                  ["TenTuschersinglecell/model2012.pyx"],
                  include_dirs=[np.get_include()]),
        ]

except ImportError:
    print("WARNING: Cython not found, not creating cython modules.")
    ext_packages = None
    build_ext = None

# note that simple .py files are already included because the plugin folders
# are packages.
extra_packages = {}
#extra_packages.update(find_packages('pythonanalyser.plugins.filter'))

# Add to the package list.
packages.extend(extra_packages.keys())

package_data = {
    # Extra data which is not a python file can go here.
    }

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# Writing full information:
def write_version_py(filename='TenTuschersinglecell/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('bondarenkoclass/version.py'):
        # must be a source distribution, use existing version file
        try:
            from bondarenkoclass.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "bondarenkoclass/version.py and the build "
                              "directory before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def setup_package():
    """Does the actual setup stuff."""
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Rewrite the version file everytime
    write_version_py()

    try:
        setup(
            name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            requires=requires,
            extras_require=extras_require,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            version=VERSION,
            platforms=PLATFORMS,
            packages=packages,
            package_data=package_data,
            #configuration=configuration
            # For Extension modules (cython)
            cmdclass={'build_ext': build_ext},
            ext_modules=ext_packages,
            )
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return


if __name__ == '__main__':
    setup_package()
