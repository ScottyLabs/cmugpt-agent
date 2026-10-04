{ pkgs, inputs, ... }:
{
  imports = [ inputs.scottylabs.devenvModules.default ];

  scottylabs = {
    enable = true;
    project.name = "cmugpt-agent";
    secrets.enable = true;
    # Local Postgres + pgvector for durable user memory. devenv creates the
    # database and exports DATABASE_URL into the shell. The agent runs
    # CREATE EXTENSION vector on setup.
    postgres = {
      enable = true;
      extensions = e: [
        e.pg_uuidv7
        e.pgvector
      ];
    };
    python.enable = true;

    kennel.services.agent = {
      customDomain = "api.cmugpt-agent.scottylabs.org";
    };
  };

  cachix.enable = false;

  # semgrep 1.172.0 only accepts pyjwt 2.13.x and the pinned nixpkgs ships
  # 2.14.0, so semgrep and the devenv shell fail to build. Skipping that
  # version check is safe: semgrep runs on 2.14.0 unchanged. Delete once
  # NixOS/nixpkgs#569851 merges and `devenv update` picks it up.
  overlays = [
    (_final: prev: {
      semgrep = prev.semgrep.overridePythonAttrs (old: {
        pythonRelaxDeps = (old.pythonRelaxDeps or [ ]) ++ [ "pyjwt" ];
      });
    })
  ];

  languages.python.package = pkgs.python312;

  processes.agent = {
    exec = "secretspec run --profile dev -- uv run cmugpt-agent";
    env.PORT = "5055";
    ready.http.get = {
      port = 5055;
      path = "/api/health";
    };
  };

  enterShell = ''
    [ -f .env ] || touch .env
  '';
}
