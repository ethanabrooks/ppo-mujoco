{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/";
    utils.url = "github:numtide/flake-utils/";
    nixgl.url = "github:guibou/nixGL";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    nixgl,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
        overlays = [nixgl.overlay];
      };
      inherit (pkgs) poetry2nix;

      python = pkgs.python39;
      overrides = pyfinal: pyprev: rec {
        miniworld = pyprev.miniworld.overridePythonAttrs (oldAttrs: {
          buildInputs = oldAttrs.buildInputs ++ [pkgs.patchelf];
          preFixup = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib pkgs.libGL pkgs.libGLU pkgs.mesa]}:$LD_LIBRARY_PATH"
          '';
        });
      };
      poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };

      myNixgl = pkgs.nixgl.override {
        nvidiaVersion = "535.86.05";
        nvidiaHash = "sha256-QH3wyjZjLr2Fj8YtpbixJP/DvM7VAzgXusnCcaI69ts=";
      };
    in {
      devShell = pkgs.mkShell {
        LD_LIBRARY_PATH = with pkgs; "${libGLU}/lib:${freetype}/lib";
        PYTHONBREAKPOINT = "ipdb.set_trace";
        buildInputs = with pkgs; [
          myNixgl.nixGLNvidia
          poetry
          poetryEnv
        ];
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
