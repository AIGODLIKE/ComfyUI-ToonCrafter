# Introduction
This project is used to enable [ToonCrafter](https://github.com/ToonCrafter/ToonCrafter) to be used in ComfyUI.

You can use it to achieve generative keyframe animation(RTX 4090,26s)

https://github.com/AIGODLIKE/ComfyUI-ToonCrafter/assets/116185401/68edb789-5a8e-418f-ae35-e3cfe6ab1300

https://github.com/AIGODLIKE/ComfyUI-ToonCrafter/assets/116185401/86553c22-9395-4b0a-9d8d-0c29c7467bd3

And use it in Blender for animation rendering and prediction

**Additionally, it can be used completely without a network**

## Installation
1. ComfyUI Custom Node
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/AIGODLIKE/ComfyUI-ToonCrafter
   cd ComfyUI-ToonCrafter
   # install dependencies
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```
2. Model Prepare
   - Download the weights:

     - [512 full weights](https://github.com/ToonCrafter/ToonCrafter?tab=readme-ov-file#-models) *High VRAM usage, fp16 reccomended*
     
     - [512 fp16 weights](https://huggingface.co/Kijai/DynamiCrafter_pruned/resolve/main/tooncrafter_512_interp-fp16.safetensors)


   - Put it in into `ComfyUI-ToonCrafter\ToonCrafter\checkpoints\tooncrafter_512_interp_v1` for example 512x512.
3. Enjoy it!

## Showcases

### Blender

You can even use it directly in Blender!([ComfyUI-BlenderAI-node](https://github.com/AIGODLIKE/ComfyUI-BlenderAI-node))

https://github.com/AIGODLIKE/ComfyUI-ToonCrafter/assets/116185401/ca8ec681-b5bc-40a1-b12a-ad185acff477

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
