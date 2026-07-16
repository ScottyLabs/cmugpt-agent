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
          langchainMcpAdapters = python.pkgs.buildPythonPackage {
            pname = "langchain-mcp-adapters";
            version = "0.3.0";
            pyproject = true;

            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/a9/1c/b179d8650d2349a342bc1fd1aab41b34154e79c7fc86fc42bdf0bb110d6f/langchain_mcp_adapters-0.3.0.tar.gz";
              hash = "sha256-+myUlwFesoB95dDDQaNuHSRFzsuuH0ok6SL8W5Txo2w=";
            };

            build-system = [ python.pkgs.hatchling ];
            dependencies = with python.pkgs; [
              langchain-core
              mcp
              typing-extensions
            ];
            pythonRelaxDeps = [
              "langchain-core"
              "mcp"
            ];
            pythonImportsCheck = [ "langchain_mcp_adapters" ];
          };
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
            langchain-core
            langchain-openai
            langgraph
            langgraph-checkpoint-postgres
            psycopg
            langchainMcpAdapters
          ];

          # The uv environment uses psycopg's binary extra for portable local
          # setup. Nix provides libpq through its psycopg package instead.
          pythonRemoveDeps = [ "psycopg-binary" ];

          # Let the build sandbox check imports natively to prove it works
          pythonImportsCheck = [ "src.main" "agent" ];

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
