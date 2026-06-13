{ pkgs, config, inputs, ... }:
{
  imports = [ inputs.scottylabs.devenvModules.default ];

  nixpkgs.overlays = [ inputs.self.overlays.default ];

  scottylabs = {
    enable = true;
    project.name = "cmugpt-agent";
    secrets.enable = false;
  };
  scottylabs.kennel.services.api = {
    customDomain = "api.cmugpt-agent.scottylabs.org";
    oidc.redirectPaths = [ "/oauth2/callback" ];
  };

  scottylabs.postgres.enable = true;

  processes.agent.exec = "${pkgs.cmugptAgent}/bin/cmugpt-agent";

  secrets = {
      enable = true;
      vaultEndpoint = "https://secrets2.scottylabs.org";
    };
}
