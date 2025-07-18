#### flow
- stage 1 - find layer
	- ablate hook - how ablation contributes to final generation
- stage 2 - collect activations and train SAE
- stage 3 - substitute activation with SAE's output
==FOR VERIFICATION, AM I RIGHT?==

#### takeaways
- structure and flow of model
- locations of ==ATT_BLOCK== between 1D, 2D and 3D tracks

#### from entrypoint to meat
1. run_inference.py 
   - `main(conf)`
	   - `sampler_selector(conf)`
2. utils.py
   - `sampler_selector(conf)`
	   - `SelfConditioning(conf)`
3. model_runners.py
   - `Sampler` (generic sampler)
   - `SelfConditioning(conf)` 
	   - `sample_step(selfm *, t, x_t ....)`
		   - *hook for collecting activations goes goes here* ==<---==
		   - `self.model(...)`
			   - `self.load_model(self)`
				   - `model=RoseTTAFoldModule(self._conf)`


#### `RoseTTAFoldModel`
- common variables - breaking down 3 tracks:
	- 1D
		- `msa`, `msa_full`
	- 2D
		- `pair` - residue pair embeddings
	- 3D
		- `R_in, T_in` - rotation and translation matrices
		- `xyz` - 3d coordinates of atom
			- `rbf_feat`: Ca-Ca distance feature calculated from xyz coordinates
		- `alpha` - torsion angles
- key components
	- embeddings
		- `MSA_emb` and  `Extra_embd`
			- **out: embeddings for 1D and 2D and state** tracks, for given token
			- **in: msa**
			- no attention
			- produces:
				- `msa_latent` - msa embedding
				- `pair` - initial residue pair embeddings
			- extra for sequence, emb for idx
		- `Templ_emb`
			- ==out: embeddings for 2D and state :
			- ==in: emebeddings for 2D and 3D coords and angles==
			- attentions
				- `self.attn` - `Q: pair; K: templ; V: templ) 
					- where `templ` is internal embedding for 1D, 2D tracks and 3d coordinates
					- ==ATT_BLOCK: blending 3D coordinates into pair embedding)==
				- `self.attn_tor` - `Q: state; K: t1d_alpha; V: t1d_alpha) 
					- where `t1d_apha` in internal embedding for 1D track and angle features
					- ==ATT_BLOCK: blending 3D angles into pair embedding)==
			- produces: `pair, state`
			- as input: `t1d, t2d, alpha_t, xyz_t, pair, state`
			- embeddings for structural templates
	- `IterativeSimulator`
		- integrates 1D, 2D, 3D tracks using SE(3) transformations
		- updates 3D coordinates
		- consists of:
			- n `extra_block` - `IterBlock` with `d_hidden_msa=8`, `use_global_attn=True`
			- m `main_block` - `IterBlock` with `use_global_attn=False`
			- `str_refiner` - `Str2Str`
			- projection layers between `SE3_param_full['l0_out_features'], SE3_param_topk['l0_out_features']`
		- building block `IterBlock` ==*meat is here!*==
			- `msa, pair, R_in, T_in, xyz, state` -> `msa, pair, R, T, state, alpha`
			- flow: apply each block in order as below
			- consists of:
				1.  `MSAPairStr2MSA` 
					- `msa, pair, rbf_feat, state` -> `msa` (1D, 2D, 3D -> 1D)
					- `self.row_attn(msa, pair)`, `self.col_attn(msa)`  
						- ==ATT_BLOCK: 2 blocks for blending 3 tracks into msa embedding)==
				2. `MSA2Pair`
					- `msa, pair` -> `pair` (1D, 2D -> 2D)
					- no attention
				3. `PairStr2Pair` 
					- `pair, rbf_feat` -> `pair` (2D, 3D -> 2D)
					- `self.row_attn(pair, rbf_feat)`, `self.col_attn(pair, rbf_feat)`
						- ==ATT_BLOCK: 2 blocks for blending 3D coordinates into pair embedding)==
						- bias  - guiding attention mechanism
						- axial - instead 2D features each with each only specified dimensions (some mysterious module for tensor multiplication)
				4.  `Str2Str`
					- 1D, 2D, 3D -> 3D
					- updates `Ri, Ti, state, alpha`
						- given at input them and `msa, xyz`
						- under the hood: `SE3TransformerWrapper`
					- no attention layers
	- `Recycling`
		- for self-conditioning, Update inputs with outputs from previous round

calling `IterativeSimulator` with created embeddings
```python
msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, msa_full, pair, xyz[:,:,:3], state, idx, use_checkpoint=use_checkpoint,motif_mask=is_frozen_residue)
```

```python
def forward(self, seq, msa, msa_full, pair, xyz_in, state, idx, use_checkpoint=False, motif_mask=None):
    """
    input:
       seq: query sequence (B, L)
       msa: seed MSA embeddings (B, N, L, d_msa)
       msa_full: extra MSA embeddings (B, N, L, d_msa_full)
       pair: initial residue pair embeddings (B, L, L, d_pair)
       xyz_in: initial BB coordinates (B, L, n_atom, 3)
       state: initial state features containing mixture of query seq, sidechain, accuracy info (B, L, d_state)
       idx: residue index
       motif_mask: bool tensor, True if motif position that is frozen, else False(L,) 
    """
   ```

#### Attentions overview
- `Attention`
- `AttentionWithBias`
	- bias added to attn value - guidance for attention mechanism
- `SequenceWeight`
	- attention adjusted fro msa
- `MSARowAttentionWithBias`
	- each position in sequence attends to all positions in other sequence
	- query is weighted by `SequenceWeight`
- `MSAColAttention`
	- each position attends to same position in other columns
- `MSAColGlobalAttention`
	- global query across all sequences
	- averaging: `self.to_q(msa).reshape(B, N, L, self.h, self.dim)` -> `self.to_q(msa).mean(dim=1)
- `BiasedAxialAttention`
	- core concepts introduction
		- 1D attention: each token attends to all other token in sequence
		- 2D attention: each position (i, j) attends to all positions
	- given 2D data
		- use only rows or columns (1D attention)
		- but use bias to guide mechanism with structural info
