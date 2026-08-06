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

  languages.python.package = pkgs.python312;

  processes.agent = {
    exec = "secretspec run --profile dev -- uv run python src/main.py";
    env.PORT = "5000";
    ready.http.get = {
      port = 5000;
      path = "/api/health";
    };
  };

  enterShell = ''
    [ -f .env ] || touch .env
  '';
}
