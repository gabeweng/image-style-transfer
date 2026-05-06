# What If AI Could Change the Weather in Any Photo?

*A high school-friendly guide to image style transfer, diffusion models, and what we learned building one from scratch.*

---

Have you ever wished you could see what your favorite spot would look like on a rainy night instead of a sunny afternoon? Or wanted to know how that campus building looks when it snows? We spent a semester building an AI system that does exactly that — and along the way, we learned a ton about how modern AI "sees" and "imagines" the world.

## What Is Image Style Transfer?

"Style transfer" sounds fancy, but the core idea is simple: take a photo, change how it *looks* (the lighting, the sky, the weather) while keeping everything *structurally the same* (the buildings, the trees, the walkways are all in the exact same place).

Think of it like a Snapchat filter, but instead of putting dog ears on your face, you're swapping sunny afternoon for stormy night — and doing it convincingly enough that someone might think the photo was actually taken in those conditions.

Our specific challenge: photos from the University of Pennsylvania campus. Take a picture of Locust Walk on a clear day. Can AI produce a realistic version of that same scene at night, or in the rain?

---

## Why Is This Hard?

This sounds like it should be easy. Just... make it look rainier? The problem is that "rain" isn't just blue pixels scattered on the image. Realistic rain means:

- Wet reflections on the pavement
- Darker, moodier sky
- Blurred or diffused light sources
- Puddles pooling in specific spots (that make physical sense for the scene's geometry)

If you just change colors, it looks fake immediately. The AI needs to *understand the 3D structure of the scene* to apply realistic lighting and weather effects. That's the fundamental challenge.

---

## The Secret Weapon: Diffusion Models

Modern AI image generation uses something called a **diffusion model**. Here's the intuition:

Imagine you take a beautiful photo and slowly add random noise to it — like static on an old TV — until the image is pure unrecognizable fuzz. Now imagine training a neural network to *reverse* that process: given a noisy image, predict what the slightly-less-noisy version should look like. Train on millions of images, repeat thousands of times, and the network learns an incredibly rich model of what "real photos" look like.

At generation time, you start with pure noise and run this denoising process in reverse, step by step, guided by a text description like *"a photo of College Green at night, rainy weather."* The result is a photorealistic image that matches that description — generated from scratch.

The magic is that these models (like Stable Diffusion) have already been trained on billions of internet images, so they already "know" what rain looks like, what nighttime looks like, what university buildings look like. We just need to steer them correctly.

---

## Step 1: Building Our Dataset

Before any AI could run, we needed data. We spent weeks walking around Penn's campus — 34th Street, College Green, Locust Walk, Harrison, McNeil — photographing the same spots under as many different conditions as we could find: sunny days, cloudy evenings, rainy afternoons, clear nights.

We ended up with 237 raw photos. But here's the catch: every time you walk back to the same spot, you're never in *exactly* the same position. Your phone tilts a little. You're a few steps to the left. This means the buildings in two photos of the "same" scene are slightly misaligned — and that misalignment would confuse our AI during training.

### The Alignment Problem

We needed to *warp* each photo so it lines up pixel-perfectly with a reference "anchor" photo of the same location. This is called homography estimation — mathematically finding the transformation that maps one camera viewpoint onto another.

**Attempt 1 — SIFT (Classic Computer Vision):** SIFT is a classic algorithm that finds distinctive "keypoints" in an image (corners, blobs) and matches them between two photos. It worked okay for same-condition pairs (day-to-day), but completely failed for day-to-night: a bright window during the day and a glowing window at night look nothing alike to SIFT, even though they're the same window.

**Attempt 2 — Edge Maps:** We tried extracting Canny edges (outlines of objects) before matching, hoping edges would be more stable across lighting conditions. But the algorithm fixated on tiny texture edges (individual leaves, brick texture) rather than big structural boundaries like walls and windows.

**Attempt 3 — LightGlue (the winner):** LightGlue is a deep neural network specifically trained to match features across dramatically different-looking images. It "understands" that the bright rectangle at night and the dark rectangle during the day are both windows. It got us reliable alignment across all condition pairs.

After alignment, we filtered out:
- Images where the warp was too poor (14 failed)
- Duplicate captures of the same location+condition (95 filtered)
- Images where alignment cropped away too much of the scene (20 dropped)

That left us with **108 clean, aligned images** — 89 for training, 19 for testing.

---

## Step 2: The Four AI Models We Tested

We didn't train a diffusion model from scratch (that would take weeks on expensive hardware). Instead, we used *pretrained* models that already understand the visual world and adapted them to our task.

### Model A: SD img2img (Baseline)
The simplest approach: give Stable Diffusion the source image and a text prompt describing the target condition. The model does partial denoising — it adds a little noise to the source image (controlled by a "strength" parameter of 0.55), then denoises toward the target prompt. Think of it as erasing 55% of the image and letting the AI fill it back in guided by the text.

### Model B: InstructPix2Pix
This model was specifically designed for instruction-based image editing. Instead of a description, you give it a command: *"Change this photo to nighttime with rainy weather."* It was trained on millions of (instruction → edited image) pairs, so it has a strong prior on what common edits should look like.

### Model C: ControlNet
ControlNet is a specialized architecture that adds a "control signal" to guide generation while preserving structure. We used **Canny edge maps** as our control signal: we extract the outlines of the source image (every wall, window, tree trunk), and force the AI to generate an image that matches those exact outlines. This is how we enforce that the building is still the building after the transformation — the AI can't move walls.

### Model D: ControlNet + LoRA (Our Best)
Same as Model C, but we added **LoRA fine-tuning** (Low-Rank Adaptation). LoRA is a technique for adapting a large pretrained model to a specific domain using very little data. We fine-tuned a small set of extra weights on our 108 Penn campus images, teaching the model what Penn buildings actually look like. The LoRA adds only a fraction of new parameters on top of Stable Diffusion's 860 million — so it's efficient and doesn't "forget" the original model's capabilities.

---

## Step 3: How We Measured Success

Two metrics:

**LPIPS (Learned Perceptual Image Patch Similarity):** This measures how different the generated image is from the actual ground-truth photo of the target scene. Lower is better. It uses a neural network (not just pixel math) to measure "perceptual" similarity — matching how humans judge image quality.

**Condition Accuracy:** We trained a second neural network (a ResNet-18 classifier with two "heads" — one for time-of-day, one for weather) to look at a generated image and predict what conditions it represents. If we asked for "night + rainy" and the classifier says "night + rainy," that counts as correct. Higher is better.

---

## What We Found

| Model | LPIPS (lower = better) | Time-of-Day Accuracy | Weather Accuracy |
|---|---|---|---|
| SD Baseline | 0.623 | 22.7% | 40.9% |
| InstructPix2Pix | 0.647 | **77.3%** | 18.2% |
| ControlNet | 0.627 | 22.7% | 18.2% |
| **ControlNet + LoRA** | **0.620** | 36.4% | **40.9%** |

The results show a fascinating tension: **no single model wins on everything.**

- **InstructPix2Pix** is dramatically better at conveying time-of-day (77%!). It really "gets" that night means dark skies and streetlights. But it moves the scene around too much — the LPIPS score (0.647) is the worst, meaning the generated image drifts further from the actual ground-truth photo.

- **ControlNet + LoRA** is the best at preserving the scene (lowest LPIPS, 0.620). The Canny edges act like a skeleton that keeps the AI from "moving the furniture." Fine-tuning with LoRA on Penn images gave an extra bump, improving time-of-day accuracy from 22.7% to 36.4% compared to base ControlNet.

- **All LPIPS scores are above 0.6**, which is still pretty far from a perfect score of 0. This reflects the fundamental difficulty: generating a new weather condition convincingly enough to match the actual pixel values of a real photo taken in that condition is an unsolved problem.

---

## The Big Takeaway: Fidelity vs. Creativity

The core lesson: **there is a fundamental tradeoff between how well the AI follows your instructions and how faithfully it preserves the original scene.**

InstructPix2Pix is like a very creative painter — you say "make it night" and it paints a beautiful, convincing night scene. But it takes some creative liberties with the composition. ControlNet with LoRA is like a careful architect — it keeps every wall in the right place, but it's more conservative about what "nighttime" actually means.

For real-world applications like architecture previews (show me what this building looks like after dark) or movie pre-visualization (what does this street look like in a storm?), you'd choose different models depending on how much structural precision you need.

---

## What We'd Do Differently

1. **More data.** 108 images is tiny by machine learning standards. With 1,000+ campus images covering more conditions, the LoRA fine-tuning would almost certainly work much better.

2. **Paired training data.** The ideal dataset has *pairs* of the exact same scene under different conditions. Ours are close but not perfectly paired (slight timing and position differences). True paired data would unlock much stronger training signals.

3. **Human evaluation.** LPIPS and condition accuracy are imperfect proxies. Ultimately, the question "does this look realistic?" is best answered by showing images to real people and asking them.

---

## Why Should You Care?

Image style transfer might seem like a fun toy, but it has real applications:

- **Architecture and urban planning:** Show clients what a building looks like in different seasons or at different times of day, without building it first.
- **Video games and film:** Automatically generate night/weather variants of scenes to reduce production time.
- **Climate visualization:** Show what a city might look like under different climate scenarios.

And fundamentally, the challenge of "change the appearance but not the structure" touches on deep questions about how AI represents the world — separating *what is there* from *how it looks* is something humans do effortlessly but AI is still learning.

---

## Want to Learn More?

If you're interested in exploring this yourself, here are some starting points:

- **Stable Diffusion** — The foundational model we built on (open source!)
- **Hugging Face Diffusers** — The Python library that makes these models accessible in a few lines of code
- **LightGlue** — The neural matcher we used for alignment ([github.com/cvg/lightglue](https://github.com/cvg/lightglue))
- **LoRA paper** — "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al. (the technique applies to image models too)

You don't need a supercomputer to get started — many of these models run for free in Google Colab, and the Hugging Face ecosystem makes it surprisingly accessible to experiment. The hardest part, as we found, isn't the models — it's the data.

---

*Written by Gabe Weng, Tomi Adenekan, and Ren Tao as part of CIS 4190/5190: Applied Machine Learning at the University of Pennsylvania, Spring 2026.*
