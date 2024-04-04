# Downloads the declared version of libjemalloc source code and builds it.

project(jemalloc_faiss)

include(ExternalProject)

SET (_target_name "jemalloc_faiss")

SET (_external_target_name "jemalloc_faiss_external")

SET (_build_script "${CMAKE_CURRENT_SOURCE_DIR}/jemalloc_unix.sh")

SET (_jemalloc_src_location "${CMAKE_CURRENT_BINARY_DIR}/src/${_external_target_name}")

SET (_jemalloc_install_location "${CMAKE_CURRENT_BINARY_DIR}/install")

ExternalProject_Add(${_external_target_name}
  GIT_REPOSITORY        https://github.com/jemalloc/jemalloc
  GIT_TAG               5.3.0
  PREFIX                "${CMAKE_CURRENT_BINARY_DIR}"
  CONFIGURE_COMMAND     ""
  BUILD_COMMAND         "${_build_script}" ${_jemalloc_src_location} ${_jemalloc_install_location}
  INSTALL_COMMAND       ""
)

set (_jemalloc_include ${_jemalloc_install_location}/include)
set (_jemalloc_lib ${_jemalloc_install_location}/lib)

add_library(${_target_name} SHARED IMPORTED GLOBAL)
set_target_properties(${_target_name} PROPERTIES IMPORTED_LOCATION ${_jemalloc_lib}/libjemalloc_faiss.so)

install(FILES
  ${_jemalloc_lib}/libjemalloc_faiss.so
  ${_jemalloc_lib}/libjemalloc_faiss.so.2
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY
  ${_jemalloc_include}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)