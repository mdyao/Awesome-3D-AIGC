# Awesome 3D AIGC Resources

A curated list of papers and open-source resources focused on 3D AIGC, intended to keep pace with the anticipated surge of research in the coming months. If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.


## Table of contents

- [Survey](#survey)
- [Text to 3D](#text-to-3d)
- [Image to 3D](#image-to-3d)
- [Editing](#editing)
- [Efficient-3D-AIGC](#diffusion)
- [Multi-modality to 3D](multi-modality-to-3D)
- [Avatar Generation](#avatars)
- [Human Body Generation](#human-body-generation)
- [Autonomous Driving](#autonomous-driving)
- [BioMedical](biomedical)
- [Misc](#misc)

- [Open Source Implementations](#open-source-implementations)
  * [Reference](#reference)
  * [Unofficial Implementations](#unofficial-implementations)
  * [Datasets](#datasets)
  * [Other](#other)
- [Blog Posts](#blog-posts)
- [Tutorial Videos](#tutorial-videos)
- [Credits](#credits)


<details span>
<summary><b>Update Log:</b></summary>
<br>


  **Dec 27, 2023**: Initial list with first 6 papers.

</details>

<be>

## Survey

### 1. Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era

**Authors**: Chenghao Li, Chaoning Zhang, Atish Waghwase, Lik-Hang Lee, Francois Rameau, Yang Yang, Sung-Ho Bae, Choong Seon Hong

<details span>
<summary><b>Abstract</b></summary>
Generative AI (AIGC, a.k.a. AI generated content) has made remarkable progress in the past few years, among which text-guided content generation is the most practical one since it enables the interaction between human instruction and AIGC. Due to the development in text-to-image as well 3D modeling technologies (like NeRF), text-to-3D has become a newly emerging yet highly active research field. Our work conducts the first yet comprehensive survey on text-to-3D to help readers interested in this direction quickly catch up with its fast development. First, we introduce 3D data representations, including both Euclidean data and non-Euclidean data. On top of that, we introduce various foundation technologies as well as summarize how recent works combine those foundation technologies to realize satisfactory text-to-3D. Moreover, we summarize how text-to-3D technology is used in various applications, including avatar generation, texture generation, shape transformation, and scene generation.
</details>

  [📄 Paper](https://arxiv.org/pdf/2305.06131.pdf) 

### 2. Deep Generative Models on 3D Representations: A Survey

**Authors**: Zifan Shi, Sida Peng, Yinghao Xu, Andreas Geiger, Yiyi Liao, Yujun Shen

<details span>
<summary><b>Abstract</b></summary>
Generative models aim to learn the distribution of observed data by generating new instances. With the advent of neural networks, deep generative models, including variational autoencoders (VAEs), generative adversarial networks (GANs), and diffusion models (DMs), have progressed remarkably in synthesizing 2D images. Recently, researchers started to shift focus from 2D to 3D space, considering that 3D data is more closely aligned with our physical world and holds immense practical potential. However, unlike 2D images, which possess an inherent and efficient representation (\textit{i.e.}, a pixel grid), representing 3D data poses significantly greater challenges. Ideally, a robust 3D representation should be capable of accurately modeling complex shapes and appearances while being highly efficient in handling high-resolution data with high processing speeds and low memory requirements. Regrettably, existing 3D representations, such as point clouds, meshes, and neural fields, often fail to satisfy all of these requirements simultaneously. In this survey, we thoroughly review the ongoing developments of 3D generative models, including methods that employ 2D and 3D supervision. Our analysis centers on generative models, with a particular focus on the representations utilized in this context. We believe our survey will help the community to track the field's evolution and to spark innovative ideas to propel progress towards solving this challenging task.
</details>

  [📄 Paper](https://arxiv.org/pdf/2210.15663.pdf) | [🌐 Project Page](https://github.com/justimyhxu/awesome-3D-generation/)

### 3. A survey of deep learning-based 3D shape generation
**Authors**: Qun-Ce Xu, Tai-Jiang Mu, Yong-Liang Yang 

<details span>
<summary><b>Abstract</b></summary>
Deep learning has been successfully used for tasks in the 2D image domain. Research on 3D computer vision and deep geometry learning has also attracted attention. Considerable achievements have been made regarding feature extraction and discrimination of 3D shapes. Following recent advances in deep generative models such as generative adversarial networks, effective generation of 3D shapes has become an active research topic. Unlike 2D images with a regular grid structure, 3D shapes have various representations, such as voxels, point clouds, meshes, and implicit functions. For deep learning of 3D shapes, shape representation has to be taken into account as there is no unified representation that can cover all tasks well. Factors such as the representativeness of geometry and topology often largely affect the quality of the generated 3D shapes. In this survey, we comprehensively review works on deep-learning-based 3D shape generation by classifying and discussing them in terms of the underlying shape representation and the architecture of the shape generator. The advantages and disadvantages of each class are further analyzed. We also consider the 3D shape datasets commonly used for shape generation. Finally, we present several potential research directions that hopefully can inspire future works on this topic.
</details>

  [📄 Paper](https://link.springer.com/article/10.1007/s41095-022-0321-5) 


## Text to 3D

### 1. DreamFusion: Text-to-3D using 2D Diffusion

**Authors**: Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

<details span>
<summary><b>Abstract</b></summary>
Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D assets and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.
</details>

  [📄 Paper](https://arxiv.org/abs/2209.14988) | [🌐 Project Page](https://dreamfusion3d.github.io) | [💻 Code](https://github.com/ashawkey/stable-dreamfusion) 

### 2. Zero-1-to-3: Zero-shot One Image to 3D Object


**Authors**: Ruoshi Liu1, Rundi Wu1, Basile Van Hoorick1, Pavel Tokmakov2, Sergey Zakharov2, Carl Vondrick1

<details span>
<summary><b>Abstract</b></summary>
We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.
</details>

  [📄 Paper](https://arxiv.org/abs/2303.11328) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/cvlab-columbia/zero123) | [🤗 Hugging Face](https://huggingface.co/spaces/cvlab/zero123-live)

### 3. Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model

**Authors**: Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, Hao Su

<details span>
<summary><b>Abstract</b></summary>
We report Zero123++, an image-conditioned diffusion model for generating 3D-consistent multi-view images from a single input view. To take full advantage of pretrained 2D generative priors, we develop various conditioning and training schemes to minimize the effort of finetuning from off-the-shelf image diffusion models such as Stable Diffusion. Zero123++ excels in producing high-quality, consistent multi-view images from a single image, overcoming common issues like texture degradation and geometric misalignment. Furthermore, we showcase the feasibility of training a ControlNet on Zero123++ for enhanced control over the generation process. The code is available at this https URL.
</details>

  [📄 Paper](https://arxiv.org/abs/2310.15110) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/SUDO-AI-3D/zero123plus) | [🤗 Hugging Face](https://huggingface.co/spaces/sudo-ai/zero123plus-demo-space)



## Image to 3D

## Multi-modality to 3D

## Efficient-3D-AIGC

## Avatar Generation

## Human Body Generation

## Autonomous Driving



## BioMedical

### 1. MinD-3D: Reconstruct High-quality 3D objects in Human Brain
**Authors**: Jianxiong Gao, Yuqian Fu, Yun Wang, Xuelin Qian, Jianfeng Feng, Yanwei Fu

<details span>
<summary><b>Abstract</b></summary>
In this paper, we introduce Recon3DMind, a groundbreaking task focused on reconstructing 3D visuals from Functional Magnetic Resonance Imaging (fMRI) signals. This represents a major step forward in cognitive neuroscience and computer vision. To support this task, we present the fMRI-Shape dataset, utilizing 360-degree view videos of 3D objects for comprehensive fMRI signal capture. Containing 55 categories of common objects from daily life, this dataset will bolster future research endeavors. We also propose MinD-3D, a novel and effective three-stage framework that decodes and reconstructs the brain's 3D visual information from fMRI signals. This method starts by extracting and aggregating features from fMRI frames using a neuro-fusion encoder, then employs a feature bridge diffusion model to generate corresponding visual features, and ultimately recovers the 3D object through a generative transformer decoder. Our experiments demonstrate that this method effectively extracts features that are valid and highly correlated with visual regions of interest (ROIs) in fMRI signals. Notably, it not only reconstructs 3D objects with high semantic relevance and spatial similarity but also significantly deepens our understanding of the human brain's 3D visual processing capabilities. Project page at:  https://jianxgao.github.io/MinD-3D.
</details>

  [📄 Paper](https://arxiv.org/abs/2312.07485) | [🌐 Project Page](https://jianxgao.github.io/MinD-3D/) | [💻 Code](https://github.com/JianxGao/MinD-3D) 


## Misc









## Credits

- Thanks to [MrNeRF](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) for inspiring me to construct this repo.
