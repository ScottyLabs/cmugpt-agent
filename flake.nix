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
    in
    {
      packages = forAllSystems (
        system:
        lib.optionalAttrs (system == "x86_64-linux") (
          let
            pkgs = pkgsFor system;
            python = pkgs.python312;

            cmugptAgent = python.pkgs.buildPythonApplication {
              pname = "cmugpt-agent";
              version = (lib.importTOML ./pyproject.toml).project.version;
              pyproject = true;
              src = ./.;

              nativeBuildInputs = (with pkgs; [ makeWrapper ]) ++ (with python.pkgs; [ hatchling ]);
              propagatedBuildInputs = with python.pkgs; [
                flask
                httpx
                mcp
                openai
                pydantic
                python-dotenv
              ];

              pythonImportsCheck = [ "agent" ];

              postPatch = ''
                substituteInPlace src/main.py --replace-fail 'from pathlib import Path' $'import os\nfrom pathlib import Path'
                substituteInPlace src/main.py --replace-fail 'app.run(host="0.0.0.0", port=8000, debug=True)' \
                  'app.run(host=os.environ.get("CMUGPT_HOST", "127.0.0.1"), port=int(os.environ.get("CMUGPT_PORT", "8000")), debug=os.environ.get("CMUGPT_DEBUG", "").lower() in ("1", "true"))'
              '';

              postInstall = ''
                mkdir -p $out/share/cmugpt-agent
                cp -r "$src/src" $out/share/cmugpt-agent/
                makeWrapper ${python}/bin/python $out/bin/cmugpt-agent \
                  --set PYTHONNOUSERSITE 1 \
                  --prefix PYTHONPATH : "$out/lib/python${python.pythonVersion}/site-packages:$out/share/cmugpt-agent/src" \
                  --add-flags "$out/share/cmugpt-agent/src/main.py"
              '';

              meta.mainProgram = "cmugpt-agent";
            };
          in
          {
            inherit cmugptAgent;
            default = cmugptAgent;
          }
        )
      );
    };
}
