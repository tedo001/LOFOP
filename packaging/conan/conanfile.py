"""Conan 2.x recipe for the LOFOP C++ inference SDK.

Like vcpkg, Conan does not host code: this recipe describes how to fetch and
build a tagged release. Use it to publish to your own remote::

    conan create packaging/conan --version 1.3.0
    conan upload lofop/1.3.0 -r <your-remote> --confirm

To submit to ConanCenter instead, this file becomes
``recipes/lofop/all/conanfile.py`` in a pull request to
``conan-io/conan-center-index``, alongside a ``config.yml`` mapping versions to
source URLs and hashes.
"""

import os

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get


class LofopConan(ConanFile):
    name = "lofop"
    package_type = "library"
    license = "Apache-2.0"
    homepage = "https://github.com/tedo001/LOFOP"
    url = "https://github.com/tedo001/LOFOP"
    description = (
        "C++ inference SDK for LOFOP-Detect object detection models. Runs an "
        "exported ONNX model with no Python or PyTorch in the deployed artifact."
    )
    topics = ("computer-vision", "object-detection", "onnx", "inference", "deep-learning")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        # Off by default so the package builds with no third-party dependency;
        # graph execution is then supplied through the Engine interface.
        "with_onnxruntime": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_onnxruntime": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        # The CMake project lives in cpp/ within the source tree.
        cmake_layout(self, src_folder="src")

    def requirements(self):
        if self.options.with_onnxruntime:
            self.requires("onnxruntime/1.18.1")

    def validate(self):
        from conan.tools.build import check_min_cppstd

        check_min_cppstd(self, 17)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["LOFOP_WITH_ONNXRUNTIME"] = bool(
            self.options.with_onnxruntime
        )
        toolchain.cache_variables["LOFOP_BUILD_TESTS"] = False
        toolchain.cache_variables["LOFOP_BUILD_EXAMPLES"] = False
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_script_folder="cpp")
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["lofop"]
        # Consumers use the same target name the upstream CMake package exports.
        self.cpp_info.set_property("cmake_file_name", "lofop")
        self.cpp_info.set_property("cmake_target_name", "lofop::lofop")
        if self.options.with_onnxruntime:
            self.cpp_info.defines.append("LOFOP_WITH_ONNXRUNTIME")
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs.append("m")
