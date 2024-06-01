# Introduction
This project is used to enable [ToonCrafter](https://github.com/ToonCrafter/ToonCrafter) to be used in ComfyUI.

You can use it to achieve generative keyframe animation

And use it in Blender for animation rendering and prediction

## Installation
1. ComfyUI Custom Node
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/AIGODLIKE/ComfyUI-ToonCrafter
   cd ComfyUI-ToonCrafter
   # install dependencies
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```
2. Model Prepare [see here](https://github.com/ToonCrafter/ToonCrafter?tab=readme-ov-file#-models).
   - Download the model.ckpt
   - Put it in into `ComfyUI-ToonCrafter\ToonCrafter\checkpoints\tooncrafter_512_interp_v1` for example 512x512.
3. Enjoy it!

## Showcases

Directly used for Blender's rendering output

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Generated video</td>
    </tr>
  <tr>
  <td>
    <img src=https://github.com/AIGODLIKE/ComfyUI-ToonCrafter/assets/116185401/1f4a4fe6-52ff-45f8-9a88-277a4eee9c8c width="250">
  </td>
  <td>
    <img src=https://github.com/AIGODLIKE/ComfyUI-ToonCrafter/assets/116185401/cf7c1d18-33a4-45e6-bc9a-9f7dc53b0547 width="250">
  </td>
  <td>
    <img src=https://github.com/AIGODLIKE/ComfyUI-ToonCrafter/assets/116185401/9a10f89b-e515-44db-869d-1769ae7d9677 width="250">
  </td>
  </tr>
</table>
