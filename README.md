# LTX-2 Web Studio

A full-stack, GPU-accelerated video generation studio specifically engineered for the [Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) foundational video model. 

LTX-2 Web Studio provides an unparalleled, professional-grade interface for AI video synthesis, boasting native integration with PyTorch, infinite canvas workspaces, and dynamic tensor-mapping for `FP8` execution on local consumer hardware.

---

## ⚡ Features

* **Infinite Canvas Workspace**: Experience a node-style, infinitely pannable and zoomable (`Ctrl+Scroll`) viewport explicitly designed replacing rigid layout boxes natively with `@panzoom/panzoom`.
* **Zero-Constraint Dynamic UI**: Seamlessly switch between rendering architectures (e.g., 1-Stage, 2-Stage, IC LoRAs). UI components, parameters, and defaults dynamically build straight from natively installed definitions (`ltx_pipelines`). Parameters map across mode switches with zero data loss.
* **Aggressive On-The-Fly FP8 Quantization**: Can't fit the colossal 42GB `BF16` weights on your 24GB RTX 3090/4090? LTX-2 Web Studio dynamically utilizes PyTorch's native `FP8 Cast` to explicitly halve linear multiplier loads natively in VRAM during execution flawlessly.
* **Decoupled Base Weight Management**: Natively inject Custom LoRAs alongside standard HuggingFace/CivitAI checkpoints. Our `ModelManager` actively sweeps and parses models (hiding default distillation dependencies so you can switch architectures completely safely without double-stacking).
* **Smart Dimension Math**: Slide your desired seconds and input FPS—the React-style layout engine strictly auto-clamps and converts outputs flawlessly locking aspect ratios exactly matching the 32-pixel dimensional divisibility strictly enforced by the PyTorch backend.
* **Auto-Downloading API System**: Paste HuggingFace Hub or CivitAI API links sequentially inside `config.yaml`. The backend detects missing safetensors on boot and securely fetches them instantly.

---

## 🚀 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd LTX-2
   ```

2. **Configure your Environments**
   Populate your `.env` securely if using private downloads or gated huggingface models:
   ```env
   HF_TOKEN=hf_xxxxxx...
   CIVITAI_TOKEN=xxx...
   ```

3. **Modify `config.yaml` to specify model URLs**
   The application securely fetches what it needs directly.

4. **Boot the Super-Server**
   ```bash
   ./start.sh
   ```
   > this bash script explicitly initializes the FastAPI server securely on an open development port alongside the Vite hot-module frontend securely! Open up your local host to access the Web Studio UI.

---

## 🧩 Architectural Layout (3-Panel Mode)

Our application runs entirely un-collapsed optimally.
*   **Left-Panel [Workflow Logic]**: Pipeline mode switches, Core Prompts, Negatives, Execution, Configs and Dimension settings cleanly mapped dynamically.
*   **Center-Panel [Viewport]**: Purely Infinite Canvas mapped to zoom tracking seamlessly integrating raw HTML5 `<video>` output nodes natively.
*   **Right-Panel [LoRAs Stack]**: Solely designed to stack custom ComfyUI standard format `.safetensors`. Switch between LoRAs safely avoiding the base distillation automatically.

---

## 📜 Legal
Distributed under standard Open Source licensing. Uses `ltx_pipelines` directly natively provided by Lightricks parameters. 
