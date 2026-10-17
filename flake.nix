{
  description = "kiseki: development shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.python314 pkgs.uv pkgs.nodejs_24 pkgs.electron pkgs.rsync pkgs.openssh ];
        # uv-managed wheels (JAX, CUDA) find libcuda through the NixOS driver path.
        shellHook = ''
          export UV_PYTHON=${pkgs.python314}/bin/python3.14
          export ELECTRON_BIN=${pkgs.electron}/bin/electron
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          fi
        '';
      };
    };
}
