# demo

demo
The cleanest way is: the multi-input condition encoder produces extra features that are injected into the frozen SDXL UNet at every denoising step.

For our setup, I’d recommend this specific design:

SDXL still runs as img2img
The rendered target RGB is the anchor image.
It is encoded by the SDXL VAE into the starting latent.
Noise is added according to the img2img strength.
The frozen SDXL UNet denoises that latent as usual.
The condition encoder reads the other inputs separately
target render depth
target render alpha / trust mask
warped neighbor 0: RGB, depth, visibility
warped neighbor 1: RGB, depth, visibility
warped neighbor 2: RGB, depth, visibility
Because we chose fixed nearest slots, these can stay in fixed channels/branches.

The encoder outputs multi-scale feature maps
one set aligned to the UNet down blocks
one for the mid block
optionally one for the up blocks
So this is similar in spirit to ControlNet, except:

not one control image
not one shared encoder
instead a small custom encoder for our multimodal inputs
Those features are injected as residuals into the frozen UNet
at each block, we add a learned residual tensor to the hidden state
usually through zero-initialized 1x1 or small conv projection layers
only these projection layers + the condition encoder are trainable
So mathematically it is like:

h_block = frozen_unet_block(h_block, text, timestep) + cond_residual_block
Optional extra pooled tokens
Besides spatial feature maps, the condition encoder can also produce a small pooled summary vector
that vector can be projected into extra conditioning tokens for cross-attention
but I would treat this as optional
the main signal should be the spatial residual maps
Why this is a good fit:

the base SDXL model stays frozen
the trainable part mostly learns how to use the conditions
the rendered target image still anchors layout/viewpoint
the warped neighbors/depth/visibility tell the model where missing detail should come from
