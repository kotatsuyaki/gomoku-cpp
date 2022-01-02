{
  inputs = {
    nixpkgs.url = github:nixos/nixpkgs/nixos-21.11;
    utils.url = github:numtide/flake-utils;
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            # nix dev
            rnix-lsp
            nixpkgs-fmt
            # C++ dev
            clang_13
            lld_13
            lldb_13
            mold
            (clang-tools.override {
              llvmPackages = pkgs.llvmPackages_13;
            })
            cmake
            gnumake
            bear
            # C++ deps
            (libtorch-bin.override {
              cudaSupport = true;
            })
            fmt
            boost
            cudatoolkit_11_4
            cudnn_cudatoolkit_11_4
          ];
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit_11_4}";
          EXTRA_LDFLAGS = "-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
          EXTRA_CCFLAGS = "-I/usr/include";
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:$LD_LIBRARY_PATH";
        };
      });
}

