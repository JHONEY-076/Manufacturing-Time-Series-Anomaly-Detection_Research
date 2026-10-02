# Diffusion figure

- diffusion_overview.drawio: editable native rectangles, labels, arrows, and waveform polylines; no embedded raster images.
- diffusion_overview.svg: vector preview for publication.
- diffusion_overview.png: 4500 × 1650 px preview at 300 dpi.

Open the .drawio file in diagrams.net. Each state is grouped; ungroup to edit individual waveforms and labels. Text and connectors elsewhere can be edited directly.

The diffusion-step notation s and S matches section_3_1_preliminaries.md, Section 3.1.2. The waveforms are schematic, not experimental results. The two directions show a forward Gaussian-noising chain and learned reverse transitions. Reverse generation does not guarantee recovery of the exact original sample. The terminal noise state illustrates the usual sufficiently-noised DDPM limit; it is not a claim about a particular finite schedule in the FMD implementation. The generic noise predictor is architecture-neutral.

Suggested caption: Forward diffusion and learned reverse denoising for time-series data. Gaussian noise is progressively added to the data, while a learned noise predictor guides the reverse process.

Reference: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. https://arxiv.org/abs/2006.11239
