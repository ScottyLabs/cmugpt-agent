{
  description = "Bark Agent";

  nixConfig = {
    extra-substituters = [ "https://scottylabs.cachix.org" ];
    extra-trusted-public-keys = [
      "scottylabs.cachix.org-1:hajjEX5SLi/Y7yYloiXTt2IOr3towcTGRhMh1vu6Tjg="
    ];
  };

  inputs = {
    scottylabs.url = "git+https://git.cmu.dev/ScottyLabs/kennel";
    nixpkgs.follows = "scottylabs/nixpkgs";
    flake-parts.follows = "scottylabs/flake-parts";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.scottylabs.flakeModules.default ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        {
          pkgs,
          lib,
          helpers,
          ...
        }:
        {
          scottylabs = {
            project.name = "cmugpt-agent";

            # the agent runs CREATE EXTENSION vector on setup
            postgres = {
              enable = true;
              extensions = e: [
                e.pg_uuidv7
                e.pgvector
              ];
            };

            python = {
              enable = true;
              package = pkgs.python312;
            };

            processes.agent = {
              run = "PORT=5055 exec secretspec run -P dev -- uv run cmugpt-agent";
              ready = "${lib.getExe pkgs.curl} -sf http://localhost:5055/api/health";
              depends = [ "postgres-setup" ];
            };

            shell.hook = ''
              [ -f "$SCOTTYLABS_ROOT/.env" ] || touch "$SCOTTYLABS_ROOT/.env"
            '';
          };

          services.agent = {
            package = lib.addMetaAttrs { mainProgram = "cmugpt-agent"; } (
              helpers.buildPythonService {
                src = ./.;
                python = pkgs.python312;
              }
            );
            customDomain = "api.cmugpt-agent.scottylabs.org";
          };
        };
    };
}
