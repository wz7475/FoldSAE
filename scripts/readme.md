## tips
- **run all script from root of main repo**
- convenient way to pass python path for scripts is `$(which python)` in shell with activated environment
## structure
- `envs` - scripts to set up environment for given model
  - create and activate environment with python version given in comment (use any provider you like e.g. conda, uv, pdm, pyenv etc.)
  - install dependencies by run `bash <paht_to_script>` _sometimes confirming by typing `y` may be necessary_
- `rfdiffsae` - RFDiffusion
- `protein-struct-pipe` - ProteinMPNN and bio_embeddings
- `clean` - enzyme classifiers
- `seqeunce_diversity` - clustering with CDHIT _currently conda is required for this scripts_
- `managment` - scripts for moving files and directories