{ pkgs, inputs, ... }:
{
  imports = [ inputs.scottylabs.devenvModules.default ];

  scottylabs = {
    enable = true;
    project.name = "cmugpt-agent";
    secrets.enable = true;
    # Durable user memory (agent/memory.py) is backed by Postgres + pgvector.
    # Enabling this boots postgresql_18 in devenv, creates the `cmugpt-agent`
    # database, and auto-exports DATABASE_URL (a unix-socket conn string) into
    # the shell - the agent picks that up and langgraph runs `CREATE EXTENSION
    # vector` on setup. We add pgvector to the module's default extensions.
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
