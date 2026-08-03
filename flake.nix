{
  description = "CMUGPT Agent";

  nixConfig = {
    extra-substituters = [ "https://scottylabs.cachix.org" ];
    extra-trusted-public-keys = [
      "scottylabs.cachix.org-1:hajjEX5SLi/Y7yYloiXTt2IOr3towcTGRhMh1vu6Tjg="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        nixpkgs.follows = "nixpkgs";
      };
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs = {
        pyproject-nix.follows = "pyproject-nix";
        uv2nix.follows = "uv2nix";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = lib.genAttrs supportedSystems;

      # The whole dependency graph (including this project itself) resolved
      # straight from uv.lock, so it can never drift out of sync the way a
      # hand-maintained propagatedBuildInputs list did.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; }).overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
          ]
        )
      );

      mkCmugptAgent =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          venv = pythonSets.${system}.mkVirtualEnv "cmugpt-agent-env" workspace.deps.default;
        in
        pkgs.runCommand "cmugpt-agent" { meta.mainProgram = "cmugpt-agent"; } ''
          mkdir -p $out/bin
          ln -s ${venv}/bin/cmugpt-agent $out/bin/cmugpt-agent
        '';
    in
    {
      packages = forAllSystems (
        system:
        let
          cmugptAgent = mkCmugptAgent system;
        in
        {
          inherit cmugptAgent;
          agent = cmugptAgent;
          default = cmugptAgent;
        }
      );
    };
}
