# vcpkg port for the LOFOP C++ inference SDK.
#
# vcpkg does not host code: this recipe downloads the tagged source tarball
# that GitHub generates for a release and verifies it against SHA512. When
# cutting a new version, bump `version` in vcpkg.json and refresh the hash:
#
#   curl -sL https://github.com/tedo001/LOFOP/archive/refs/tags/v1.3.0.tar.gz | sha512sum
#
# (Or set SHA512 to 64 zero bytes and run the install once -- vcpkg prints the
# actual hash in the mismatch error.)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO tedo001/LOFOP
    REF "v${VERSION}"
    SHA512 0
    HEAD_REF main
)

vcpkg_check_features(
    OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        onnxruntime LOFOP_WITH_ONNXRUNTIME
)

# The CMake project lives in cpp/ and compiles the shared kernels from
# ../lofop/csrc, which the source tarball also contains.
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/cpp"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DLOFOP_BUILD_TESTS=OFF
        -DLOFOP_BUILD_EXAMPLES=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME lofop CONFIG_PATH lib/cmake/lofop)
vcpkg_copy_pdbs()

# Headers ship once, from the release tree.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
