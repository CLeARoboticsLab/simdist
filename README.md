<div align="center">

# Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation

<p>
  <a href="https://jake-levy.github.io/">Jacob Levy</a><sup>*1</sup>, <a href="https://tyler-westenbroek.github.io/">Tyler Westenbroek</a><sup>*2</sup>,
  <a href="https://kevinhuang8.github.io/">Kevin Huang</a><sup>2</sup>, <a href="https://palafox.info/">Fernando Palafox</a><sup>1</sup>,
  <a href="https://patrickyin.me/">Patrick Yin</a><sup>2</sup>, <a href="https://www.linkedin.com/in/shayegan/">Shayegan Omidshafiei</a><sup>3</sup>,
  <a href="https://dkkim93.github.io/">Dong-Ki Kim</a><sup>3</sup>, <a href="https://homes.cs.washington.edu/~abhgupta/">Abhishek Gupta</a><sup>2</sup>,
  <a href="https://dfridovi.github.io/">David Fridovich-Keil</a><sup>1</sup>
  <br>
  <span><sup>1</sup> UT Austin</span>&emsp;
  <span><sup>2</sup> UW</span>&emsp;
  <span><sup>3</sup> FieldAI</span>&emsp;
  <span><sup>*</sup> Equal Contribution</span>
</p>

[![Website](docs/assets/badge-website.svg)](https://sim-dist.github.io/)
[![Paper](docs/assets/badge-pdf.svg)](#)
</div>

<!-- ##  -->

This project implements **Simulation Distillation (SimDist)**, a scalable framework the distills structural priors from a simulator into a latent world model and enables rapid real-world adaptation via online planning and supervised dynamics finetuning.

<div align="center">
  <video
    src="docs/assets/hero-background-desktop.mp4"
    style="width: 100%; max-width: 800px; height: auto;"
    controls autoplay loop muted>
  </video>
</div>

## 📚 Usage

- [💾 Installation](docs/installation.md)
- [🧠 Pretraining (Go2)](docs/pretraining_go2.md)
  - [Expert Policy Training](docs/pretraining_go2.md#expert-policy-training)
  - [Data Generation](docs/pretraining_go2.md#data-generation)
  - [World Model Pretraining](docs/pretraining_go2.md#world-model-pretraining)
  - [Deployment (Simulation)](docs/pretraining_go2.md#deployment-simulation)
- [🤖 Deployment (Real-World Go2)](docs/deployment_go2.md)
  - [Hardware Setup](docs/deployment_go2.md#hardware-setup)
  - [Simulation Setup (Optional)](docs/deployment_go2.md#simulation-setup-optional)
  - [Startup](docs/deployment_go2.md#startup)
  - [Running the Robot](docs/deployment_go2.md#running-the-robot)
  - [Shutdown](docs/deployment_go2.md#shutdown)
- [⚙️ Adaptation](docs/adaptation.md)
  - [Processing Real-World Data](docs/adaptation.md#processing-real-world-data)
  - [Finetuning the World Model](docs/adaptation.md#finetuning-the-world-model)

## Acknowledgements

For SLAM, We use the version of ["point_lio_unilidar"](https://github.com/unitreerobotics/point_lio_unilidar) from [`autonomy_stack_go2`](https://github.com/jizhang-cmu/autonomy_stack_go2) from @jizhang-cmu.

## Citation

If you find our work helpful, please cite:

```bibtex
@InProceedings{levy2026simdist,
  title={Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation},
  author={Jacob Levy and Tyler Westenbroek and Kevin Huang and Fernando Palafox and Patrick Yin and Shayegan Omidshafiei and Dong-Ki Kim and Abhishek Gupta and David Fridovich-Keil},
  year={2026}
}
