Here is a technical, ready-to-use `README.md` for your repository.

---

# Vision Transformer (ViT) for Autonomous UAV Perception

A custom Vision Transformer (ViT) architecture engineered specifically for autonomous Unmanned Aerial Vehicles (UAVs). This project implements advanced regularization, self-supervised pre-training, and uncertainty quantification to ensure robust perception in unpredictable aerial environments (e.g., sensor occlusion, domain shift, and noise).

Currently trained on CIFAR-10 as a proxy dataset for aerial/ground target classification (planes, ships, cars).

## 🧠 Core Architecture

Unlike standard ViTs, this model implements custom structural modifications for edge robotics:

* **Convolutional Stem (`ConvPatchEmbed`):** Replaces standard linear patch projection with a `3x3 Conv2d` layer. Injects spatial inductive biases to rapidly extract local features (critical for detecting small obstacles at a distance).
* **Stochastic Depth (`DropPath`):** Randomly drops entire residual block pathways during training, forcing the network to build highly redundant, distributed representations to survive partial sensor failures.
* **Pre-Norm Layering:** Utilizes `LayerNorm` prior to Attention and MLP blocks for stable training gradients.

## ⚙️ Training Pipeline

The training process is split into a two-phase curriculum to maximize data efficiency and robustness.

### Phase 1: Self-Supervised Pre-training (Masked Autoencoder)

* **Mechanism:** Wraps the ViT encoder in an MAE framework. Randomly masks **75%** of input image patches.
* **Objective:** Forces the encoder to reconstruct missing pixels using a lightweight linear decoder.
* **Result:** The model learns underlying geometric structures and spatial relationships without relying on labels, making it highly resilient to occluded sensors (e.g., mud or glare on the camera).

### Phase 2: Supervised Fine-Tuning (SAM Optimizer)

* **Mechanism:** Discards the MAE decoder, attaches a linear classification head, and applies `CutMix` and `MixUp` augmentations via a custom collate function.
* **Optimization:** Utilizes **Sharpness-Aware Minimization (SAM)**.
* *Gradient Ascent Step:* Perturbs weights to the point of maximum local loss ($\rho$).
* *Gradient Descent Step:* Minimizes the loss from that worst-case position.


* **Result:** Forces the network to settle in wide, "flat" minima. Guarantees better generalization against environmental domain shifts (lighting changes, weather).

## 🛡️ Safety, Explainability, & Evaluation

Production-grade UAVs require auditability. This codebase includes built-in analytics:

* **Attention Rollout:** Extracts raw attention weights across all transformer layers to generate a global heatmap for the `[CLS]` token. Visually maps exactly which pixels drove the UAV's classification decision.
* **Epistemic Uncertainty (MC Dropout):** Re-enables dropout during inference and runs 50 forward passes per image. Calculates predictive variance and entropy. If the UAV encounters severe degradation (e.g., thick fog), entropy spikes, allowing the system to trigger a human-in-the-loop fallback.
* **Degradation Testing:** Automated pipeline to inject clamping and Gaussian noise to empirically validate the MC Dropout safety bounds.
* **Latent Manifold Projection:** t-SNE integration to visualize the 192-dimensional latent feature space in 2D.

## 🚀 Edge Deployment (ONNX)

The model is configured for immediate edge deployment (e.g., Nvidia Jetson Nano/Orin) via TensorRT:

* **Format:** ONNX (`opset_version=14`)
* **Optimizations:** Constant folding enabled to pre-calculate static operations.
* **Flexibility:** Dynamic batch axis configured for multi-camera stream processing.

## 📦 Requirements

```text
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
seaborn
scikit-learn
opencv-python

```

## 🛠️ Quick Start

1. Clone the repository and install dependencies.
2. Ensure CUDA is available for Mixed Precision (`torch.amp.GradScaler`) training.
3. Run the notebook/script sequentially to trigger the MAE pre-training, SAM fine-tuning, and ONNX export.
4. Output model will be saved as `uav_vit_model_v3.onnx`.
