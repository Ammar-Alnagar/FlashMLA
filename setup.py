import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).resolve().parent

# Stable dispatcher API shared by both build platforms. The fused kernels live
# in their own translation unit.
API_SOURCES = [
    "csrc/api/api.cpp",
    "csrc/api/sparse_prefill.cpp",
    "csrc/api/sparse_decode.cpp",
    "csrc/api/dense_fwd.cpp",
    "csrc/api/dense_bwd.cpp",
    "csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp",
]

# The extension modules, one per build platform
EXT_NAME_CUDA = "flash_mla.cuda"
EXT_NAME_ASCEND = "flash_mla.npu"

# `-gencode` targets. Includes sm_89 (Ada Lovelace) and sm_100a / sm_103a (Blackwell).
# Can be overridden via TORCH_CUDA_ARCH_LIST or FLASH_MLA_CUDA_ARCHS environment variables.
CUDA_ARCHS = ["sm_89", "sm_100a", "sm_103a"]

# Build against PyTorch's stable C++ ABI.
STABLE_ABI_FLAGS = [
    "-DTORCH_TARGET_VERSION=0x020a000000000000",  # PyTorch >= 2.10
]
CUDA_STABLE_ABI_FLAGS = [*STABLE_ABI_FLAGS, "-DUSE_CUDA"]

# Head counts (H_Q) the fused core_attn kernel is instantiated for: 128 (2-CTA cluster) and 64
HEADS_FOR_FUSED_CORE_ATTN = (64, 128)

# The sm100 (Blackwell) sources use tcgen05 / TMA / cluster features that only exist from sm_100
# onwards, and pulling in CUTLASS 4.x for them dominates the build time. They are therefore only
# compiled when at least one Blackwell target is requested, so that a pure sm_89 (Ada) build stays
# lean. Everything sm_89 needs is architecture-independent C++ plus the sm89 kernels.
CUDA_KERNEL_SOURCES_SM89 = [
    "csrc/cuda_kernels/sm89/permute/permute_q_b_proj.cu",
    "csrc/cuda_kernels/sm89/permute/permute_wv_proj.cu",
    "csrc/cuda_kernels/sm89/sparse/sparse_prefill_sm89.cu",
    "csrc/cuda_kernels/sm89/sparse/sparse_decode_sm89.cu",
    "csrc/cuda_kernels/sm89/sparse/fused_norm_rope_attn_rope_cast_sm89.cu",
    "csrc/cuda_kernels/sm89/dense/dense_sm89.cu",
]

CUDA_KERNEL_SOURCES_SMXX = [
    # Small kernels for decoding: tile scheduler metadata generation and the split-KV combine
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",
]

CUDA_KERNEL_SOURCES_SM100 = [
    # sm100 dense prefill (cutlass FMHA): forward and backward
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",

    # sm100 sparse prefill
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",

    # sm100 sparse prefill for small topk, and head128 decoding
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu",

    # sm100 sparse decode (head 64, with and without split-KV, on the fp8 and the fp4 cache)
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu",

    # Fused Norm + RoPE + Core Attn + RoPE + Cast: 6 instantiations per head count
    # (V4.1 fp8 prefill, V4.1 fp8 decode and V4.1 fp4 decode, each with and without q norm)
    *[f"csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/{model}_h{h}_{mode}_{norm}.cu"
      for h in HEADS_FOR_FUSED_CORE_ATTN
      for model, mode in (("v41", "prefill"), ("v41", "decode"), ("v41fp4", "decode"))
      for norm in ("norm", "nonorm")],
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu",
]

# The Ascend instantiations are explicit template instantiations of `run_sparse_fwd_kernel`.
# Kept for: h64 x {prefill, decode} x {V4.1 fp8, V4.1 fp4 (decode only)} x {sink, nosink}
ASCEND_KERNEL_SOURCES = [
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_prefill_nosink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_prefill_sink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_decode_nosink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41_h64_decode_sink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41fp4_h64_decode_nosink.asc",
    "csrc/ascend_kernels/prefill/sparse/instantiations/v41fp4_h64_decode_sink.asc",
]

def resolve_cuda_archs():
    """The list of `sm_*` targets this build targets, after applying the env overrides."""
    archs = CUDA_ARCHS
    env_archs = os.getenv("TORCH_CUDA_ARCH_LIST") or os.getenv("FLASH_MLA_CUDA_ARCHS")
    if env_archs:
        # e.g., "8.9" or "8.9 10.0"
        arch_list = []
        for a in env_archs.replace(";", " ").split():
            a_clean = a.strip().replace(".", "")
            if a_clean:
                if not a_clean.startswith("sm_"):
                    a_clean = f"sm_{a_clean}"
                arch_list.append(a_clean)
        if arch_list:
            archs = arch_list
    return archs


def is_sm100_arch(arch: str) -> bool:
    """Whether `arch` is a Blackwell target, i.e. compute capability 10.x or higher."""
    compute_arch = arch.removeprefix("sm_").rstrip("af")
    return int(compute_arch) >= 100


def get_cuda_arch_flags(archs):
    arch_flags = []
    for arch in archs:
        compute_arch = arch.removeprefix("sm_")
        arch_flags.extend(["-gencode", f"arch=compute_{compute_arch},code={arch}"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


# `flash_mla/__init__.py` needs to know which platform this build targets and the exact file
# name of the extension (which depends on the CPython ABI tag), so the build records both in a
# generated module instead of having the package guess at import time.
BUILD_INFO_PATH = Path("flash_mla") / "_build_info.py"


def write_build_info(backend: str, extension_filename: str, build_lib: Optional[str] = None) -> None:
    contents = (
        '"""Generated by setup.py at build time. Do not edit."""\n'
        "\n"
        f'BACKEND = "{backend}"\n'
        f'EXTENSION = "{extension_filename}"\n'
    )
    targets = [THIS_DIR / BUILD_INFO_PATH]
    if build_lib is not None:
        targets.append(Path(build_lib) / BUILD_INFO_PATH)
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)


class WriteBuildInfoMixin:
    """Record the built platform and extension file name after every build."""

    def run(self):
        super().run()
        if self.dry_run:
            return
        for ext in self.extensions:
            backend = ext.name.rsplit(".", 1)[-1]
            extension_filename = os.path.basename(self.get_ext_filename(ext.name))
            # `build_lib` is where wheels are packaged from; also write into the source tree so
            # that `build_ext --inplace` leaves an importable package behind.
            build_lib = self.build_lib if os.path.isdir(self.build_lib) else None
            write_build_info(backend, extension_filename, build_lib)


def build_on_cuda_platform():
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        CUDA_HOME,
    )

    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable.
    # For more details, see `torch/utils/cpp_extension.py`
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    nvcc_version = subprocess.check_output(
        [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    nvcc_version_number = nvcc_version.split("release ")[1].split(",")[0].strip()
    major, minor = map(int, nvcc_version_number.split("."))
    print(f"Compiling using NVCC {major}.{minor}")
    assert major > 13 or (major == 13 and minor >= 1), \
        "Compiling the kernels requires NVCC 13.1 or higher (cvt with .scaled::n2::ue8m0)."

    archs = resolve_cuda_archs()
    build_sm100 = any(is_sm100_arch(arch) for arch in archs)
    # Programmatic dependent launch needs both producer and consumer on sm_90+ (see
    # csrc/cuda_kernels/smxx/decode/combine/combine.cu).
    enable_pdl = any(int(arch.removeprefix("sm_").rstrip("af")) >= 90 for arch in archs)
    pdl_define = ["-DFLASH_MLA_ENABLE_PDL"] if enable_pdl else []
    cuda_kernel_sources = [
        *CUDA_KERNEL_SOURCES_SM89,
        *CUDA_KERNEL_SOURCES_SMXX,
        *(CUDA_KERNEL_SOURCES_SM100 if build_sm100 else []),
    ]
    print(f"Building for: {', '.join(archs)}")
    print("  sm89 (Ada Lovelace) kernels: enabled")
    print(f"  sm100/sm103 (Blackwell) kernels: {'enabled' if build_sm100 else 'disabled (set TORCH_CUDA_ARCH_LIST=8.9 for an sm89-only build)'}")

    class BuildExtensionWithPostActions(WriteBuildInfoMixin, BuildExtension):
        """
        Record the build info, then run the register-spill check after building.
        """
        def run(self):
            super().run()

            if not self.dry_run:
                from tests.kernelkit import check_kernel_reg_spill_in_artifact

                so_paths = [
                    os.path.join(self.build_lib, self.get_ext_filename(EXT_NAME_CUDA)),
                    self.get_ext_fullpath(EXT_NAME_CUDA),    # May point to the extension in the source tree for editable installs
                ]
                so_paths = list(set(so_paths))  # deduplicate
                assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
                for so_path in so_paths:
                    spilled_kernels = check_kernel_reg_spill_in_artifact(
                        so_path, quiet=True, suppress_checking_env_var="FLASH_MLA_SKIP_REG_SPILL_CHECK"
                    )
                    # The restored upstream CUTLASS FMHA kernels (cutlass::device_kernel<...>) keep
                    # their stack frames, so they are exempt. Only the kernels defined by this
                    # repository have to be spill-free.
                    own_spilled_kernels = [name for name in spilled_kernels if not name.startswith("_ZN7cutlass")]
                    num_exempt_kernels = len(spilled_kernels) - len(own_spilled_kernels)
                    if num_exempt_kernels > 0:
                        print(f"{num_exempt_kernels} upstream CUTLASS kernel(s) keep their stack frames (exempt)")
                    if len(own_spilled_kernels) > 0:
                        for kernel_name in own_spilled_kernels:
                            print(f"  {kernel_name}")
                        print("Register spilling detected. Build failed!")
                        sys.exit(1)

    ext_modules = [
        CUDAExtension(
            name=EXT_NAME_CUDA,
            sources=[
                *API_SOURCES,
                *cuda_kernel_sources,
            ],
            extra_compile_args={
                # The host-side (`.cpp`) translation units are not compiled by nvcc, so `__CUDACC__`
                # is not defined and neither kerutils' nor our own platform switch is set automatically.
                "cxx": [
                    *CUDA_STABLE_ABI_FLAGS,
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Wno-deprecated-declarations",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                    "-DKERUTILS_IS_BUILD_ON_CUDA",
                ] + (["-DFLASH_MLA_IS_BUILD_ON_SM100"] if build_sm100 else []),
                "nvcc": get_nvcc_thread_args() + CUDA_STABLE_ABI_FLAGS + [
                    "-O3",
                    "-std=c++20",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                    "-lineinfo",
                    "--source-in-ptx",
                    "--diag-suppress=445",  # "constant "N" is not used in declaring the parameter types of function template"
                    "-DNDEBUG",
                    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
                ] + (["-DFLASH_MLA_IS_BUILD_ON_SM100"] if build_sm100 else []) + pdl_define + get_cuda_arch_flags(archs),
            },
            include_dirs=[
                THIS_DIR / "csrc",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "tools" / "util" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
                Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",    # for cuda/std headers in CUDA 13+
                Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
            ],
            # On a machine without the NVIDIA driver (e.g. a CPU-only build node) `-lcuda` needs
            # the stub libraries that ship with the toolkit
            extra_link_args=[
                f"-L{Path(CUDA_HOME) / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'}",
                f"-L{Path(CUDA_HOME) / 'targets' / 'sbsa-linux' / 'lib' / 'stubs'}",
                "-lcuda",
            ],
            # Registration uses STABLE_TORCH_LIBRARY and does not need the
            # Python C API, so one extension can serve every CPython >= 3.10.
            py_limited_api=True,
        )
    ]

    return ext_modules, BuildExtensionWithPostActions


def build_on_ascend_platform():
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    import torch_npu

    arch = os.environ.get("ASCEND_NPU_ARCH", "dav-3510")
    asc_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    if not os.path.isdir(asc_home):
        raise RuntimeError(
            f"Ascend toolkit not found at '{asc_home}'. "
            "Please set ASCEND_HOME_PATH to the root of your CANN installation "
            "(e.g. /usr/local/Ascend/cann-9.1.0)."
        )

    class AscendBuildExtension(WriteBuildInfoMixin, BuildExtension):
        def build_extensions(self):
            self.compiler.src_extensions += [".asc"]
            os.environ["CXX"] = "bisheng"
            os.environ["CC"] = "bisheng"
            super().build_extensions()

    ext_modules = [
        CppExtension(
            name=EXT_NAME_ASCEND,
            sources=[
                *API_SOURCES,
                "csrc/api/ascend_stream.cpp",
                *ASCEND_KERNEL_SOURCES,
            ],
            extra_compile_args={
                "cxx": [
                    *STABLE_ABI_FLAGS,
                    "-Ofast",
                    "-std=c++20",
                    f"--npu-arch={arch}",
                    "-DFLASH_MLA_IS_BUILD_ON_ASCEND",
                    "-isystem", str(Path(asc_home) / "aarch64-linux" / "include"),  # `-isystem` suppresses warnings
                    "-isystem", str(Path(asc_home) / "aarch64-linux" / "asc" / "include"),
                    "-isystem", str(Path(torch_npu.__file__).parent / "include"),
                    "-mllvm", "-enable-hiipu-vf-loop-unroll",
                    "--cce-res-usage",
                    # Otherwise some `__simd_callee__` are incorrectly marked as "function 'XXX' is not needed and will not be emitted"
                    "-Wno-unneeded-internal-declaration",
                    "-Wno-microsoft-template",
                    "-Wno-deprecated-declarations",
                    # Disable argument preload at the beginning of kernels to avoid a hardware bug
                    "-mllvm", "-cce-aicore-dcpreload-args=false",
                ],
            },
            extra_link_args=[
                str(Path(asc_home) / "aarch64-linux" / "lib64" / "libascendc_runtime.a"),
            ],
            include_dirs=[
                THIS_DIR / "csrc",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
            ],
            libraries=[
                "torch_npu",
                "ascendcl",
            ],
            library_dirs=[
                str(Path(torch_npu.__file__).parent / "lib"),
                str(Path(asc_home) / "aarch64-linux" / "lib64"),
            ],
            # Registration uses STABLE_TORCH_LIBRARY and does not need the
            # Python C API, so one extension can serve every CPython >= 3.10.
            py_limited_api=True,
        )
    ]

    return ext_modules, AscendBuildExtension


def get_build_target_platform() -> str:
    """
    The target platform of the build, "CUDA" or "ASCEND".

    `FLASH_MLA_BUILD_TARGET_PLATFORM` takes precedence. Otherwise the platform is detected
    without any dependency probing: the Ascend driver exposes `/dev/davinci_manager`,
    so its absence means CUDA.
    """
    overrided_platform = os.environ.get("FLASH_MLA_BUILD_TARGET_PLATFORM")
    if overrided_platform is not None:
        available = ["CUDA", "ASCEND"]
        if overrided_platform not in available:
            raise ValueError(
                f"Invalid `FLASH_MLA_BUILD_TARGET_PLATFORM`: {overrided_platform}. "
                f"Available values are {available}"
            )
        return overrided_platform
    return "ASCEND" if os.path.exists("/dev/davinci_manager") else "CUDA"


def get_version() -> str:
    try:
        rev = "+" + subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=THIS_DIR, stderr=subprocess.DEVNULL
        ).decode("ascii").rstrip()
    except Exception:
        rev = ""
    return "1.0.0" + rev


def main():
    current_platform = get_build_target_platform()
    print(f"Build target: {current_platform}")

    if current_platform == "CUDA":
        ext_modules, build_ext = build_on_cuda_platform()
    else:
        ext_modules, build_ext = build_on_ascend_platform()

    setup(
        name="flash_mla",
        version=get_version(),
        packages=find_packages(include=["flash_mla"]),
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )


if __name__ == "__main__":
    main()
