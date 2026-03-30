# demo

demo


Split the problem into two different edit regimes instead of treating the whole image as one diffusion task:

Trusted-geometry region: the 3DGS render is geometrically correct, so the goal is appearance transfer only from the warped views while preserving the render’s structure.
Low-alpha region: the 3DGS render has weak support or holes, so this is a true inpainting/completion problem.
The diffusion model should therefore receive:

the 3DGS render as the geometry anchor
warped views as appearance references
an explicit region decomposition telling it where to do:
appearance refinement
inpainting
no change
The key change from the previous plan is: do not let diffusion rewrite trusted geometry freely. Outside low-alpha areas, the model should behave like a guided appearance harmonizer, not a generator.

Key Changes
Region decomposition

Derive three masks from render_alpha.png and warped-view visibility:
keep_mask: very high-trust geometry, default alpha >= 0.98
transfer_mask: geometry-trusted but appearance-improvable region, default 0.60 <= alpha < 0.98
inpaint_mask: low-support region, default alpha < 0.60
Dilate transfer_mask by 4 px and inpaint_mask by 8 px.
Resolve overlaps by priority:
inpaint_mask first
then transfer_mask
then keep_mask
Two-stage target input construction

Build appearance_prefill_rgb.png:
start from render_rgb.png
inside transfer_mask, bring in appearance from warped neighbors, but only where neighbor visibility is valid
do not replace the whole region with a single donor; blend warped-view color toward the render using confidence-weighted mixing
Build inpaint_prefill_rgb.png:
start from appearance_prefill_rgb.png
inside inpaint_mask, use donor warps opportunistically where visible
leave unresolved pixels as the original render for diffusion to complete
Save both masks separately:
transfer_mask.png
inpaint_mask.png
keep_mask.png
Appearance transfer rule in trusted geometry

In transfer_mask, preserve geometry from the render and only borrow appearance from the 3 warped views.
For each pixel, compute donor weights from:
warped visibility
inverse camera-center distance
local alpha confidence
local RGB agreement to the render
Use the donor blend to form a soft appearance hint, not a hard overwrite.
Final transfer-prefill formula:
prefill = (1 - beta) * render + beta * donor_blend
default beta range: 0.25 to 0.60, increasing as alpha decreases
This keeps the render’s geometry dominant while still importing better texture/color.
Inpainting rule in low-alpha region

Treat inpaint_mask as true completion.
If warped donors exist, use them as partial hints.
If no donors exist, leave the render as-is and let diffusion infer the missing content.
In this region, diffusion is allowed to modify the image much more aggressively than in transfer_mask.
Model route

Use SDXL inpainting LoRA as the primary model:
base: diffusers/stable-diffusion-xl-1.0-inpainting-0.1
Keep the pretrained inpainting model frozen except UNet LoRA adapters.
Do not build a multi-branch encoder in the first refined version.
Instead, encode the task through:
inpaint_prefill_rgb.png
a single edit_mask.png
optional auxiliary preview image for analysis only, not as model input
Single-mask training target

Collapse the editable regions for the inpainting model into one edit_mask.png:
edit_mask = transfer_mask U inpaint_mask
But keep region labels in metadata and in the loss:
transfer_mask = gentle appearance correction
inpaint_mask = strong completion zone
Loss weighting / behavior

Weight diffusion supervision by region:
inpaint_mask: 1.0
transfer_mask: 0.5
keep_mask: 0.0
Add a boundary band weight:
0.25 on a 12 px ring around edit_mask
At inference, composite strictly:
outside edit_mask, copy pixels directly from render_rgb.png
inside transfer_mask, use the diffusion result but keep low denoise
inside inpaint_mask, use the full diffusion output
Training/inference schedule

Train on crops, but enforce region-aware sampling:
50% crops centered on inpaint_mask
30% crops centered on transfer_mask
20% random crops
Crop sizes stay:
1024x768
896x672
768x576
Validation should report:
full-image PSNR
transfer_mask PSNR
inpaint_mask PSNR
keep_mask PSNR after final compositing
Inference defaults:
transfer_mask: low denoise behavior via the same inpaint run, but geometry is protected by the prefill and final composite
inpaint_mask: normal inpaint behavior
start with 6 steps
