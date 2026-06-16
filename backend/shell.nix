{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  shellHook = ''
    if [ -d /run/opengl-driver/lib ]; then
      export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      export TRITON_LIBCUDA_PATH="''${TRITON_LIBCUDA_PATH:-/run/opengl-driver/lib}"
    fi

    if [ -z "''${SSL_CERT_FILE:-}" ] && [ -r /etc/ssl/certs/ca-certificates.crt ]; then
      export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
    fi
  '';
}
