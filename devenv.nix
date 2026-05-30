{ pkgs, config, inputs, ... }:
{
  imports = [ inputs.scottylabs.devenvModules.default ];

  nixpkgs.overlays = [ inputs.self.overlays.default ];

  scottylabs = {
    enable = true;
    project.name = "cmugpt-agent";

    kennel.services.agent = { };

    postgres.enable = true;
  };

  processes.agent = {
    exec = "${pkgs.cmugptAgent}/bin/cmugpt-agent";
    ready.http.get = {
      port = 5000;
      path = "/health";
    };
  };
}
