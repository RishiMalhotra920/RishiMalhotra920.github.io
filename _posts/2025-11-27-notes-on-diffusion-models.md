---
layout: distill
title: Notes on Diffusion Models
date: 2025-11-27 10:00:00-0800
description: Some notes on diffusion models starting with mathematical foundations.
tags: diffusion-models math
categories: ml-research
related_posts: false
toc:
  - name: Basics
  - name: Forward and Backward SDEs and ODEs
  - name: Simulating the ODEs and SDEs
  - name: Training Networks for the ODE and SDE
  - name: Conditioning on Images
  - name: SDEs
  - name: References
---

These notes are based heavily on [1].

## Basics

ODEs:

$$
\frac{d}{dt} x_t = u_t(x_t) \tag{1}
$$

$$
dx_t = u_t(x_t) dt \tag{2}
$$

SDEs:

$$
dx_t = u_t(x_t) dt + \sigma_t dW_t \tag{3}
$$

Integrating the SDE:

$$
\int_{0}^{t} dX_s = \int_{0}^t u_s(X_s) ds + \int_{0}^t \sigma_s dW_s \tag{4}
$$

The first integral is our standard Riemann Integral:

$$
\int_{0}^t u_s(X_s) ds = \lim_{n \to \infty} \sum_{i=0}^{n-1} u_{s_i}(X_{s_i}) \cdot (s_{i+1} - s_i) \tag{5}
$$

$$
= \lim_{n \to \infty} \sum_{i=0}^{n-1} u_{s_i}(X_{s_i}) \cdot \frac{t}{n} \tag{6}
$$

The second integral is an Itô integral, following Itô's calculus:

$$
\int_{0}^t \sigma_s dW_s = \lim_{n \to \infty} \sum_{i=0}^{n-1} \sigma_{s_i} \cdot (W_{s_{i+1}} - W_{s_i}) \tag{7}
$$

$$
W_{s_{i+1}} - W_{s_i} \sim \mathcal{N}(0, s_{i+1} - s_i) \tag{8}
$$

## Forward and Backward SDEs and ODEs

Convention: $p_0$ is $p_{\text{init}}$ and $p_1$ is $p_{\text{data}}$. 

Forward process: add noise, convert $p_1 \to p_0$

Backward process: remove noise, convert $p_0 \to p_1$

Forward ODE:

$$
\frac{d}{dt} X_t = u_t(X_t) \tag{9}
$$

Backward ODE:

$$
\frac{d}{dt} X_t = -u_t(X_t) \tag{10}
$$


Forward SDE:

$$
dX_t = u_t(X_t) dt + \sigma_t dW_t \tag{11}
$$

Reverse SDE:

$$
dX_t = \left[-u_t(X_t) + \sigma_t^2 \nabla \log p_t(X_t)\right] dt + \sigma_t dW_t \tag{12}
$$

Note: you don't simulate the forward ODE and the forward SDE. You just construct the noised object and run simulation on the backward ODE and backward SDE.

## Simulating the ODEs and SDEs

**Euler Method:**

Running a Taylor series expansion of the ODE gives:

$$
X_{t+\Delta t} = X_t + X'_t \Delta t + O(t^2) \tag{13}
$$

$$
X_{t+\Delta t} = X_t + u_t^{\theta}(X_t) \Delta t + O(t^2) \tag{14}
$$

Discretizing and keeping the 1st order term:

$$
X_{t+h} = X_t + h u_t^{\theta}(X_t) \tag{15}
$$

**Euler-Maruyama Method:**

Starting from the integral form of the SDE:

$$
\int_{t}^{t+\Delta t} dX_s = \int_{t}^{t+\Delta t} u_s(X_s) ds + \int_{t}^{t+\Delta t} \sigma_s dW_s \tag{16}
$$

$$
X_{t+\Delta t} = X_t + \lim_{n \to \infty} \sum_{i=0}^{n-1} u_{s_i}(X_{s_i}) \cdot \frac{\Delta t}{n} + \lim_{n \to \infty} \sum_{i=0}^{n-1} \sigma_{s_i} \cdot (W_{s_{i+1}} - W_{s_i}) \tag{17}
$$

where $s_i = t + i \cdot \frac{\Delta t}{n}$ and $s_{i+1} - s_i = \frac{\Delta t}{n}$.

For a single step, this becomes:

$$
X_{t+\Delta t} = X_t + u_t(X_t) \cdot \Delta t + \sigma_t \cdot (W_{t+\Delta t} - W_t) \tag{18}
$$

Since $W_{t+\Delta t} - W_t \sim \mathcal{N}(0, \Delta t) = \sqrt{\Delta t} \cdot \mathcal{N}(0, 1)$, we can write:

$$
X_{t+\Delta t} = X_t + u_t(X_t) \cdot \Delta t + \sigma_t \cdot \sqrt{\Delta t} \cdot \epsilon \tag{19}
$$

where $\epsilon \sim \mathcal{N}(0, I)$.

The following SDE follows the Fokker-Planck equation (see equation (30) below):

$$
dX_t = \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\right] dt + \sigma_t dW_t \tag{20}
$$

Discretizing using the Euler-Maruyama method:

$$
X_{t+\Delta t} = X_t + u_t(X_t) \cdot \Delta t + \frac{\sigma_t^2}{2} \cdot s_t(X_t) \cdot \Delta t + \sigma_t \cdot \sqrt{\Delta t} \cdot \epsilon \tag{21}
$$

where you can train networks for $u_t$ and $s_t$. In practice, if you use a Gaussian representation for $X_t$, you can write $s_t$ as a function of $u_t$ (see equation (52) below) and train just one network.

The Euler method gives the ODE sampler. The Euler-Maruyama method produces the SDE sampler. Flow matching must use the ODE sampler; diffusion can use either the ODE or the SDE sampler!

## Training Networks for the ODE and SDE

I will consider the general case first, then assume the Gaussian case with linear noise schedules.

We need to minimize:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x}\left[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2\right] \tag{22}
$$

$u_t^{\text{target}}(x)$ is actually not tractable. So, we can instead construct the following **Conditional Flow Matching Loss** by conditioning on $z$ (the real data):

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)} \left[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\right] \tag{23}
$$

For ODEs we must minimize the above loss. For SDEs, we can minimize the $$\mathcal{L}_{\text{CFM}}(\theta)$$ or the $$\mathcal{L}_{\text{CSM}}$$ loss since the score function can be written in terms of the velocity and vice versa.

$$
\mathcal{L}_{\text{CSM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\|s_t^\theta(x) - \nabla \log p_t(x|z)\|^2\right] \tag{24}
$$

The proof that $$\mathcal{L}_{\text{CFM}}(\theta) = \mathcal{L}_{\text{FM}}(\theta) + C$$ (where $C$ is independent of $\theta$) can be found in Theorem 18 in the MIT notes [1].

**ODEs**

We define a probability path as an object that defines the probability of being at $X$ at time $t$. This path also shows how the probability distribution evolves over time:

$$
\{p_t(\cdot | z)\}_{t=0}^1 \tag{25}
$$

Also, define a flow:

$$
\begin{align}
\psi_t(x_0) \quad &\text{(flow)} \tag{26a}\\
\psi_0(x_0) &= x_0 \quad \text{(initial condition)} \tag{26b}\\
\frac{d}{dt} \psi_t(x_0) &= u_t(\psi_t(x_0)) \quad \text{(flow ODE)} \tag{26c}
\end{align}
$$

which defines the position $X_t$ reached as you simulate the ODE starting from $x_0$. $X_t$ is a random variable representing a point on just one trajectory. $\psi_t(x_0)$ is a function that represents position at time $t$ for a trajectory starting at $x_0$. It is defined for all positions.

The flow ODE imposes a constraint that the function $\psi_t$ must follow. Note: since $\psi_t$ is deterministic, it is not defined for SDEs. Instead SDEs define $p_t(X_t)$, the probability of being at a position $X_t$ at time $t$.

So, using the flow ODE for a Gaussian conditional probability path $p_t(\cdot\|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$:



$$
\begin{align}
\frac{d}{dt} \psi_t(x|z) &= u_t^{\text{target}}(\psi_t(x|z) | z) \tag{27}\\
\frac{d}{dt} (\alpha_t z + \beta_t \epsilon) &= u_t^{\text{target}}(\alpha_t z + \beta_t \epsilon | z) \tag{28}\\
\dot{\alpha}_t z + \dot{\beta}_t \epsilon &= u_t^{\text{target}}(\alpha_t z + \beta_t \epsilon | z) \tag{29}
\end{align}
$$

Reparametrizing $x = \alpha_t z + \beta_t \epsilon$ and solving for $\epsilon = \frac{x - \alpha_t z}{\beta_t}$:

$$
u_t^{\text{target}}(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}x \tag{30}
$$

With this, we have a way to pick any $x$, any $z$ and construct a training target for $u_t^{\text{target}}(x\|z)$ and minimize the loss function.

Some requirements on $\alpha_t$ and $\beta_t$:
1. Continuously differentiable w.r.t. $t$
2. Monotonic (for smooth progression from noise $\to$ data)
3. Boundary conditions:

$$
\begin{align}
\alpha_0 &= 0 = \beta_1 \tag{31a}\\
\alpha_1 &= 1 = \beta_0 \tag{31b}
\end{align}
$$

When $\alpha_t = t$ and $\beta_t = 1-t$ with the Gaussian conditional probability path, we have:

$$
u_t^{\text{target}}(x|z) = z - \epsilon \tag{32}
$$

You can verify by substituting $\alpha_t = t$, $\beta_t = 1-t$, $\dot{\alpha}_t = 1$, and $\dot{\beta}_t = -1$ into equation (30).

This simplifies the Conditional Flow Matching loss for Gaussian paths with this noise schedule. Many SOTA models are trained this way including SD3 and Meta Movie Gen Video:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I)} \left[\|u_t^\theta(tz+(1-t)\epsilon) - (z-\epsilon)\|^2\right] \tag{33}
$$

Notice that the target is a vector pointing from $\epsilon$ to $z$ and the input to the neural network is an interpolation between $z$ and $\epsilon$ based on $t$.

**Algorithm 1: Flow Matching Training (Gaussian CondOT path)**

```text
for each mini-batch of data:
    Sample z from the dataset
    Sample t ~ Unif[0, 1]
    Sample ε ~ N(0, I)
    Set x = tz + (1-t)ε
    Compute loss: L(θ) = ||u_t^θ(x) - (z - ε)||^2
    Update θ via gradient descent
```


### SDEs

We need to construct an SDE that follows the Fokker-Planck equation so that $X_t$ follows a probability path $p_t$.

**The Fokker-Planck Equation:**

Consider the SDE:

$$
X_0 \sim p_{\text{init}}, \quad dX_t = u_t(X_t) dt + \sigma_t dW_t \tag{34}
$$

Then $X_t$ has distribution $p_t$ for all $0 \leq t \leq 1$ if and only if the Fokker-Planck equation holds:

$$
\begin{align}
\frac{\partial p_t(x)}{\partial t} &= -\text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \quad \text{for all } x \in \mathbb{R}^d, 0 \leq t \leq 1 \tag{35}\\
\text{div}(v_t)(x) &= \sum_{i=1}^{d} \frac{\partial}{\partial x_i} v_t(x) \quad \text{(Divergence)} \tag{36}\\
\Delta p_t(x) &= \sum_{i=1}^{d} \frac{\partial^2}{\partial x_i^2} p_t(x) \quad \text{(Laplacian)} \tag{37}
\end{align}
$$

Intuitively, the divergence denotes how much flow is moving out of a unit volume over time. The Laplacian denotes the curvature of the probability distribution, e.g., if Laplacian $> 0$, probability will accumulate in that region.

Imagine particles moving over a certain location $x$ over time according to the drift term, plus some noise is added at each timestep due to the diffusion term in the SDE. Then, intuitively, the Fokker-Planck equation shows that the change in probability at a location $x$ over time $t$ is the sum of the inflow to the unit volume (drift term) plus how the noise would accumulate at that location (diffusion term).

Proof for this is present in Appendix B of the MIT notes.

Then, the SDE below follows the Fokker-Planck equation. The term in square brackets is the realization for $u_t(X_t)$ in the general SDE equation:

$$
dX_t = \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2}\nabla \log p_t(X_t)\right] dt + \sigma_t dW_t \tag{38}
$$

which implies:

$$
X_t \sim p_t \tag{39}
$$

This is very similar to the ODE case:

$$
\begin{align}
\frac{d}{dt} X_t &= u_t^{\theta}(X_t) \tag{40}\\
X_t &\sim p_t \tag{41}
\end{align}
$$

Specifically, Theorem 13 in the MIT notes says that if you construct an SDE with the form in equation (38), then $X_t$ follows the same distribution as in the much simpler ODE case.

With that, it becomes easy to compute $u_t^{\text{target}}(X_t\|z)$ and $s_t^{\text{target}}(X_t\|z)$ to estimate $u_t(X_t\|z)$ and $\nabla \log p_t(X_t\|z)$.

$u_t^{\text{target}}$ is the same as before (equation (30)) and for $s_t^{\text{target}}(x\|z)$, you take the gradient of the log of the Gaussian PDF:

$$
\begin{align}
u_t^{\text{target}}(x|z) &= \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}x \tag{42}\\
s_t^{\text{target}}(x|z) &= \nabla \log p_t(x|z) = -\frac{x-\alpha_t z}{\beta_t^2} \tag{43}
\end{align}
$$

Minimizing the $s_t^{\theta}$ term takes us to:

$$
\begin{align}
\mathcal{L}_{\text{CSM}}(\theta) &= \mathbb{E}\left[\left\|s_t^{\theta}(x) - s_t^{\text{target}}(x|z)\right\|^2\right] \tag{44}\\
&= \mathbb{E}\left[\left\|s_t^{\theta}(x) + \frac{x-\alpha_t z}{\beta_t^2}\right\|^2\right] \tag{45}\\
&= \mathbb{E}\left[\frac{1}{\beta_t^2}\left\|\beta_t s_t^{\theta}(\alpha_t z + \beta_t \epsilon) + \epsilon\right\|^2\right] \tag{46}
\end{align}
$$

We reparametrize $\epsilon_t^{\theta}$ to now estimate $-\beta_t s_t^{\theta}(x)$ and drop the $\frac{1}{\beta_t^2}$ factor because DDPM showed that with $\beta_t$ in the denominator the loss was unstable. This gives:

$$
\mathcal{L}_{\text{DDPM}}(\theta) \approx \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0,I)}\left[\left\| \epsilon_t(\alpha_t z + \beta_t \epsilon) - \epsilon\right\|^2\right] \tag{47}
$$

**Algorithm 2: Score Matching Training (Denoising Diffusion)**

```
for each mini-batch of data:
    Sample z from the dataset
    Sample t ~ Unif[0, 1]
    Sample ε ~ N(0, I)
    Set x = α_t z + β_t ε
    Compute loss: L(θ) = ||ε_t^θ(x) - ε||^2
    Update θ via gradient descent
```

where the network $\epsilon_t^{\theta}$ now estimates the noise.

Once you have learned $\epsilon_t^{\theta}$, which is related to $s_t^{\theta}$ by $\epsilon_t^{\theta}(x) = -\beta_t s_t^{\theta}(x)$, you can use the following conversion formula to find $u_t^{\theta}$:

$$
u_t^{\theta}(x) = \beta_t^2 \left(\frac{\dot{\alpha}_t}{\alpha_t} - \frac{\dot{\beta}_t}{\beta_t}\right) s_t^{\theta}(x) + \frac{\dot{\alpha}_t}{\alpha_t} x \tag{48}
$$

or equivalently in terms of the noise predictor:

$$
u_t^{\theta}(x) = -\beta_t \left(\frac{\dot{\alpha}_t}{\alpha_t} - \frac{\dot{\beta}_t}{\beta_t}\right) \epsilon_t^{\theta}(x) + \frac{\dot{\alpha}_t}{\alpha_t} x \tag{49}
$$

This is derived from Proposition 1 in the MIT notes.

Alternatively, you can also minimize $u_t^{\theta}$ directly using Algorithm 1 we discussed above.

Then use the Euler-Maruyama method (equation (21)) to generate samples.

Here is the SDE following Fokker-Planck in terms of $s_t^{\theta}$:

$$
dX_t = \left[u_t^{\theta}(X_t) + \frac{\sigma_t^2}{2}s_t^{\theta}(X_t)\right] dt + \sigma_t dW_t \tag{50}
$$

The conversion between score and velocity for Gaussian paths:

$$
s_t^{\theta}(x) = \frac{\alpha_t u_t^{\theta}(x) - \dot{\alpha}_t x}{\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t} \tag{52}
$$

Note, you can train $u_t^{\theta}$ or $s_t^{\theta}$ and then derive the other through this analytical form.

## Conditioning on Images

Same as above, but now we want to minimize:

### ODEs

$$
\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta) = \mathbb{E}_{(z,y) \sim p_{\text{data}}, t \sim \text{Unif}, x \sim p_t(\cdot|z)}\left[\|u_t^{\theta}(x | y) - u_t^{\text{target}}(x | z)\|^2\right] \tag{53}
$$

where $(y, z) \sim p_{\text{data}}$ are sampled jointly (e.g., text prompt and image).

However, it was found that during inference, images generated don't adhere well enough to the prompt. So you train $u_t$ normally but during inference you scale up the contribution from the guidance term.

The guided vector field can be decomposed as:

$$
u_t^{\text{target}}(x|y) = u_t^{\text{target}}(x) + b_t \nabla \log p_t(y|x) \tag{54}
$$

where $b_t = \beta_t^2 \left(\frac{\dot{\alpha}_t}{\alpha_t} - \frac{\dot{\beta}_t}{\beta_t}\right)$.

You can view the first term as the natural flow and the second term as a guidance signal that shows how the log probability of $y\|x$ changes w.r.t. $x$. So, as the pixels of blurry image $x$ change, how does the probability of text $y$ change. Note the sum of these two terms forms the target which is the true signal your network $u_t^{\theta}$ must approximate.

So, to increase your network's adherence to the text signal during inference, you add a weight $w > 1$:

$$
\tilde{u}_t(x|y) = u_t^{\text{target}}(x) + w \cdot b_t \nabla \log p_t(y|x) \tag{55}
$$

which can be rewritten as:

$$
\tilde{u}_t(x|y) = (1-w)u_t^{\text{target}}(x|\emptyset) + w \cdot u_t^{\text{target}}(x|y) \tag{56}
$$

where $\emptyset$ denotes an empty/null conditioning token.

**Classifier-Free Guidance Training:**

When training, you replace the label $y$ with an empty token $\emptyset$ with probability $\eta \in [0,1]$ (typically $\eta = 0.1$). This trains the model to approximate both $u_t^{\text{target}}(x\|\emptyset)$ (unconditional) and $u_t^{\text{target}}(x\|y)$ (conditional) in a single network.

**Algorithm 3: Classifier-Free Guidance Training**

```
for each mini-batch of data:
    Sample (z, y) from the dataset
    Sample t ~ Unif[0, 1]
    Sample ε ~ N(0, I)
    Set x = α_t z + β_t ε

    With probability η:
        Set y = ∅  (empty token)

    Compute loss: L(θ) = ||u_t^θ(x, y, t) - (z - ε)||^2
    Update θ via gradient descent
```

At inference time, use equation (56) with $w > 1$ (typically $w \in [1.5, 7.5]$).

## SDEs

You follow the same process as for ODEs when training the $u$ function and the $s$ function, but need to drop the label $y$ with probability $\eta$ as discussed above.

However, during inference, you need to amplify the probability $w$ for both the score function and the velocity function.

For the score function:

$$
\tilde{s}_t^{\theta}(x|y) = (1-w) \cdot s_t^{\theta}(x|\emptyset) + w \cdot s_t^{\theta}(x|y) \tag{57}
$$

or equivalently:

$$
\tilde{s}_t^{\theta}(x|y) = (1-w) \cdot \nabla \log p_t(x|\emptyset) + w \cdot \nabla \log p_t(x|y) \tag{58}
$$

Then the SDE for sampling becomes:

$$
dX_t = \left[u_t^{\theta}(X_t, y, t) + \frac{\sigma_t^2}{2}\tilde{s}_t^{\theta}(X_t|y)\right] dt + \sigma_t dW_t \tag{59}
$$

### Architectures

**U-Net:** Consists of encoders (increase channels, reduce spatial dimensions), midcoders (latent processing blocks), and decoders (reduce channels, increase spatial dimensions). Skip connections connect encoder and decoder layers at the same resolution.

**DiT (Diffusion Transformer):** Uses transformer blocks instead of convolutional layers, treating the image as a sequence of patches.

### Encoding the Guiding Variable

**When $y_{\text{raw}}$ is a class label:** Initialize and add an embedding network to $\theta$ and train the embedding for $y_{\text{raw}}$ end-to-end.

**When $y_{\text{raw}}$ is text:** Use CLIP or SigLIP embeddings or combine multiple such embeddings (e.g., CLIP for visual grounding, T5 for language understanding).

**Feeding in the embedding:** 
- Feed it into every sub-component of architecture for images. E.g., in U-Net, $y_{\text{embed}}$ and $t$ are passed to each encoder, midcoder, and decoder block.
- You can project $y$ to a shape of $C \times 1 \times 1$ and add it to each pixel in $x \in \mathbb{R}^{C \times H \times W}$.
- Alternatively, you can use cross-attention between image features and text embeddings.

### Stable Diffusion 3

For high-resolution images, you should work in the latent space of that image. SD3 trained an autoencoder (VAE) to map images to latent space and then runs flow matching in the diffusion process within this latent space.

They use:
- **CLIP embeddings** for visual grounding in images
- **T5 embeddings** for high-level understanding of language

This multi-embedding approach allows the model to leverage both visual concepts and complex language understanding.

### Meta Movie Gen Video

Introduced a **temporal autoencoder** mapping a raw video to a latent representation. The video is chopped up into spatiotemporal cubes $\to$ encoded $\to$ encodings are stitched together. The U-Net (or transformer) now predicts the velocity field or the score function in this latent space.

Key innovations:
- Temporal compression in addition to spatial compression
- Joint training on images and videos
- Scalable architecture that can handle variable-length videos

---

Written and organized by Rishi. Cleanup by Claude Sonnet.

## References

[1] P. Holderrieth and E. Erives, "An Introduction to Flow Matching and Diffusion Models," MIT Class 6.S184, 2025. Available: <a href="https://arxiv.org/pdf/2506.02070">https://arxiv.org/pdf/2506.02070</a>