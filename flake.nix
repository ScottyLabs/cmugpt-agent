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
  };

  outputs =
    { self, nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = lib.genAttrs supportedSystems;
      pkgsFor = system: nixpkgs.legacyPackages.${system};

      mkCmugptAgent = pkgs:
        let
          python = pkgs.python312;
        in
        python.pkgs.buildPythonApplication {
          pname = "cmugpt-agent";
          version = (lib.importTOML ./pyproject.toml).project.version;
          pyproject = true;
          src = ./.;

          nativeBuildInputs = with python.pkgs; [ hatchling setuptools ];

          propagatedBuildInputs = with python.pkgs; [
            fastapi
            uvicorn
            httpx
            mcp
            openai
            pydantic
            python-dotenv
          ];

          # Let the build sandbox check imports natively to prove it works
          pythonImportsCheck = [ "fastapi" "uvicorn" ];

          meta.mainProgram = "cmugpt-agent";
        };
    in
    {
      overlays.default = final: prev: {
        cmugptAgent = mkCmugptAgent final;
        agent = final.cmugptAgent;
      };

      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          cmugptAgent = mkCmugptAgent pkgs;
        in
        {
          inherit cmugptAgent;
          agent = cmugptAgent;
          default = cmugptAgent;
        }
      );
    };
}
