{ pkgs ? import <nixpkgs> {} }:
let
  fhsEnv = pkgs.buildFHSUserEnv {
    name = "fhs-env";
    targetPkgs = pkgs: (with pkgs; [
      python3
	  numactl
      cmake
      git-lfs
   ]);
   runScript = "fish";
   profile = ''
      echo "Inside profile"
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib
      source ../peft/venv/bin/activate
   '';
  };
in
pkgs.mkShell {
  shellHook = ''
    ${fhsEnv}/bin/fhs-env
  '';
}
