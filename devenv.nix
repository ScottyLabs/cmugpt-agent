{ pkgs, config, inputs, ... }:
{
  imports = [ inputs.scottylabs.devenvModules.default ];

  nixpkgs.overlays = [ inputs.self.overlays.default ];

  scottylabs = {
    enable = true;
    project.name = "cmugpt-agent";

    kennel.services.agent = {
      customDomain = "api.cmugpt-agent.scottylabs.org";
    };
  };

  processes.agent.exec = "${pkgs.cmugptAgent}/bin/cmugpt-agent";
}
