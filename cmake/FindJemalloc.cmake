# - Try to find jemalloc
# Once done this will define a target Jemalloc::jemalloc which will include all
# required definitions and libraries.

if (NOT FindJemalloc_included)

  # First try finding Jemalloc using pkg-config.
  find_package(PkgConfig QUIET)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(PC_JEMALLOC QUIET IMPORTED_TARGET GLOBAL jemalloc)

    if (PC_JEMALLOC_FOUND)
      # Ensure it created the PkgConfig target
      if(NOT TARGET PkgConfig::PC_JEMALLOC)
        message(FATAL_ERROR
          "Found Jemalloc via pkg-config, but it did not create the PkgConfig::PC_JEMALLOC target.")
      endif()

      # Make the discovered target available as Jemalloc::jemalloc
      add_library(Jemalloc::jemalloc ALIAS PkgConfig::PC_JEMALLOC)
      set(_jemalloc_found TRUE)
    endif()
  endif()

  # If that didn't find it, try finding Jemalloc using CMake's config
  # mode.
  if(NOT _jemalloc_found)
    find_package(Jemalloc CONFIG)

    if(Jemalloc_FOUND)
      if(NOT TARGET Jemalloc::jemalloc)
        # Ensure this found package created the standard target.
        message(FATAL_ERROR
          "Found Jemalloc, but it did not create the Jemalloc::jemalloc target.")
      endif()
      set(_jemalloc_found TRUE)
    endif()
  endif()

  # Determine the version of the found jemalloc.
  get_target_property(_jemalloc_include_dirs Jemalloc::jemalloc INTERFACE_INCLUDE_DIRECTORIES)
  list (GET _jemalloc_include_dirs 0 _jemalloc_include_dir)
  set(_version_regex "^#define[ \t]+JEMALLOC_VERSION[ \t]+\"([^\"]+)\".*")
  file(STRINGS "${_jemalloc_include_dir}/jemalloc/jemalloc.h"
    JEMALLOC_VERSION REGEX "${_version_regex}")
  string(REGEX REPLACE "${_version_regex}" "\\1"
    JEMALLOC_VERSION "${JEMALLOC_VERSION}")
  unset(_version_regex)

  # handle the QUIET and REQUIRED arguments, verify version (if
  # necessary), report found status, and set JEMALLOC_FOUND to TRUE.
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    Jemalloc VERSION_VAR JEMALLOC_VERSION
    REQUIRED_VARS _jemalloc_found)
  set(FindJemalloc_included TRUE)

endif (NOT FindJemalloc_included)