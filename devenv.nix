{ pkgs, inputs, lib, config, ... }:

let
  localFlake = builtins.getFlake (toString ./.);
  system = pkgs.stdenv.hostPlatform.system;
  selfPkgs = localFlake.packages.${system};
in
{
  imports = [
    inputs.scottylabs.devenvModules.default
  ];

  scottylabs = {
    enable = true;
    project.name = "cmugpt-agent";
    secrets.enable = false;
  };

  scottylabs.kennel.services.api = {
    customDomain = "api.cmugpt-agent.scottylabs.org";
  };

  # Make your shell aware of python and its dependencies directly 
  languages.python = {
    enable = true;
    package = pkgs.python312;
    poetry.enable = false;
    uv.enable = false;
  };

  # Direct packages injection into the environment to completely destroy any pathing bugs
  packages = with pkgs.python312Packages; [
    fastapi
    uvicorn
    httpx
    mcp
    openai
    pydantic
    python-dotenv
  ];

  processes.api = {
    # Run natively out of your source directory using the un-sandboxed python interpreter
    exec = "python src/main.py";
    env = {
      PORT = "8080";
    };
    ready.http.get = { port = 8080; path = "/health"; };
  };

  scottylabs.postgres.enable = true;

  processes.agent = {
    # If agent uses a different entry point (e.g. src/agent.py or main.py on different port), adjust accordingly
    exec = "python src/main.py";
    env = {
      PORT = "5000";
    };
  };

  secretspec = {
    profile = "prod";
    provider = "vault://secrets2.scottylabs.org/secret";
  };

  dotenv.disableHint = true;
}
