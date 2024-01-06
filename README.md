# Awesome 3D AIGC Resources

A curated list of papers and open-source resources focused on 3D AIGC, intended to keep pace with the anticipated surge of research in the coming months. If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.


## Table of contents

- [Survey](#survey)
- [Text to 3D Generation](#text-to-3d-generation)
- [Image to 3D Generation](#image-to-3d-generation)
- [Audio to 3D Generation](#audio-to-3d-generation)
- [3D Editing](#editing)
- [Human Avatar Generation](#human-avatar-generation)
- [City/Autonomous Driving](#autonomous-driving)
- [SLAM](#slam)
- [BioMedical](#biomedical)
- [4D AIGC](#4d-aigc)
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
<be>

  **Jan 6, 2024**: Adding recent papers. 

  **Jan 2, 2024**: Adding papers to image to 3d generation. 

  **Dec 29, 2023**: Contribute to the section on text-to-3d by adding new papers with their publication years.
  
  **Dec 27, 2023**: Initial list with first 15 papers.

</details>

<br>


## Survey:
### 1. Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era [arxiv 2023.05]

**Authors**: Chenghao Li, Chaoning Zhang, Atish Waghwase, Lik-Hang Lee, Francois Rameau, Yang Yang, Sung-Ho Bae, Choong Seon Hong

<details span>
<summary><b>Abstract</b></summary>
Generative AI (AIGC, a.k.a. AI generated content) has made remarkable progress in the past few years, among which text-guided content generation is the most practical one since it enables the interaction between human instruction and AIGC. Due to the development in text-to-image as well 3D modeling technologies (like NeRF), text-to-3D has become a newly emerging yet highly active research field. Our work conducts the first yet comprehensive survey on text-to-3D to help readers interested in this direction quickly catch up with its fast development. First, we introduce 3D data representations, including both Euclidean data and non-Euclidean data. On top of that, we introduce various foundation technologies as well as summarize how recent works combine those foundation technologies to realize satisfactory text-to-3D. Moreover, we summarize how text-to-3D technology is used in various applications, including avatar generation, texture generation, shape transformation, and scene generation.
</details>

  [📄 Paper](https://arxiv.org/pdf/2305.06131.pdf) 

### 2. Deep Generative Models on 3D Representations: A Survey [arxiv 2023.10]

**Authors**: Zifan Shi, Sida Peng, Yinghao Xu, Andreas Geiger, Yiyi Liao, Yujun Shen

<details span>
<summary><b>Abstract</b></summary>
Generative models aim to learn the distribution of observed data by generating new instances. With the advent of neural networks, deep generative models, including variational autoencoders (VAEs), generative adversarial networks (GANs), and diffusion models (DMs), have progressed remarkably in synthesizing 2D images. Recently, researchers started to shift focus from 2D to 3D space, considering that 3D data is more closely aligned with our physical world and holds immense practical potential. However, unlike 2D images, which possess an inherent and efficient representation (\textit{i.e.}, a pixel grid), representing 3D data poses significantly greater challenges. Ideally, a robust 3D representation should be capable of accurately modeling complex shapes and appearances while being highly efficient in handling high-resolution data with high processing speeds and low memory requirements. Regrettably, existing 3D representations, such as point clouds, meshes, and neural fields, often fail to satisfy all of these requirements simultaneously. In this survey, we thoroughly review the ongoing developments of 3D generative models, including methods that employ 2D and 3D supervision. Our analysis centers on generative models, with a particular focus on the representations utilized in this context. We believe our survey will help the community to track the field's evolution and to spark innovative ideas to propel progress towards solving this challenging task.
</details>

  [📄 Paper](https://arxiv.org/pdf/2210.15663.pdf) | [🌐 Project Page](https://github.com/justimyhxu/awesome-3D-generation/)

### 3. A survey of deep learning-based 3D shape generation [Computational Visual Media 2023.05]
**Authors**: Qun-Ce Xu, Tai-Jiang Mu, Yong-Liang Yang 

<details span>
<summary><b>Abstract</b></summary>
Deep learning has been successfully used for tasks in the 2D image domain. Research on 3D computer vision and deep geometry learning has also attracted attention. Considerable achievements have been made regarding feature extraction and discrimination of 3D shapes. Following recent advances in deep generative models such as generative adversarial networks, effective generation of 3D shapes has become an active research topic. Unlike 2D images with a regular grid structure, 3D shapes have various representations, such as voxels, point clouds, meshes, and implicit functions. For deep learning of 3D shapes, shape representation has to be taken into account as there is no unified representation that can cover all tasks well. Factors such as the representativeness of geometry and topology often largely affect the quality of the generated 3D shapes. In this survey, we comprehensively review works on deep-learning-based 3D shape generation by classifying and discussing them in terms of the underlying shape representation and the architecture of the shape generator. The advantages and disadvantages of each class are further analyzed. We also consider the 3D shape datasets commonly used for shape generation. Finally, we present several potential research directions that hopefully can inspire future works on this topic.
</details>

  [📄 Paper](https://link.springer.com/article/10.1007/s41095-022-0321-5) 


<br>

## Text to 3D Generation:
### 1. DreamFusion: Text-to-3D using 2D Diffusion [ICLR 2023]

**Authors**: Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

<details span>
<summary><b>Abstract</b></summary>
Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D assets and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.
</details>

  [📄 Paper](https://arxiv.org/abs/2209.14988) | [🌐 Project Page](https://dreamfusion3d.github.io) | [💻 Code](https://github.com/ashawkey/stable-dreamfusion) 



### 2. Shapeglot: Learning Language for Shape Differentiation [CVPR 2019]

**Authors**: Panos Achlioptas* Judy Fan Robert Hawkins Noah Goodman Leonidas Guibas

<details span>
<summary><b>Abstract</b></summary>
    In this work we explore how fine-grained differences between the shapes of common objects are expressed in language, grounded on images and 3D models of the objects. We first build a large scale, carefully controlled dataset of human utterances that each refers to a 2D rendering of a 3D CAD model so as to distinguish it from a set of shape-wise similar alternatives. Using this dataset, we develop neural language understanding (listening) and production (speaking) models that vary in their grounding (pure 3D forms via point-clouds vs. rendered 2D images), the degree of pragmatic reasoning captured (e.g. speakers that reason about a listener or not), and the neural architecture (e.g. with or without attention). We find models that perform well with both synthetic and human partners, and with held out utterances and objects. We also find that these models are amenable to zero-shot transfer learning to novel object classes (e.g. transfer from training on chairs to testing on lamps), as well as to real-world images drawn from furniture catalogs. Lesion studies indicate that the neural listeners depend heavily on part-related words and associate these words correctly with visual parts of objects (without any explicit network training on object parts), and that transfer to novel classes is most successful when known part-words are available. This work illustrates a practical approach to language grounding, and provides a case study in the relationship between object shape and linguistic structure when it comes to object differentiation.
</details>

  [📄 Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Achlioptas_Shapeglot_Learning_Language_for_Shape_Differentiation_ICCV_2019_paper.pdf) | [💻 Code](https://github.com/optas/shapeglot) 



### 3. Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings [ACCV 2018]

**Authors**: Kevin Chen, Christopher B. Choy, Manolis Savva, Angel X. Chang, Thomas Funkhouser, Silvio Savarese

<details span>
<summary><b>Abstract</b></summary>
    We present a method for generating colored 3D shapes from natural language. To this end, we first learn joint embeddings of freeform text descriptions and colored 3D shapes. Our model combines and extends learning by association and metric learning approaches to learn implicit cross-modal connections, and produces a joint representation that captures the many-to-many relations between language and physical properties of 3D shapes such as color and shape. To evaluate our approach, we collect a large dataset of natural language descriptions for physical 3D objects in the ShapeNet dataset. With this learned joint embedding we demonstrate text-to-shape retrieval that outperforms baseline approaches. Using our embeddings with a novel conditional Wasserstein GAN framework, we generate colored 3D shapes from text. Our method is the first to connect natural language text with realistic 3D objects exhibiting rich variations in color, texture, and shape detail. See video at this https URL
</details>

  [📄 Paper](http://arxiv.org/abs/1803.08495) | [🌐 Project Page](http://text2shape.stanford.edu/) | [💻 Code](https://github.com/kchen92/text2shape/) 

### 4. ShapeCrafter: A Recursive Text-Conditioned 3D Shape Generation Model [NeurIPS 2022]

**Authors**: Rao Fu, Xiao Zhan, Yiwen Chen, Daniel Ritchie, Srinath Sridhar

<details span>
<summary><b>Abstract</b></summary>
 We present ShapeCrafter, a neural network for recursive text-conditioned 3D shape generation. Existing methods to generate text-conditioned 3D shapes consume an entire text prompt to generate a 3D shape in a single step. However, humans tend to describe shapes recursively---we may start with an initial description and progressively add details based on intermediate results. To capture this recursive process, we introduce a method to generate a 3D shape distribution, conditioned on an initial phrase, that gradually evolves as more phrases are added. Since existing datasets are insufficient for training this approach, we present Text2Shape++, a large dataset of 369K shape--text pairs that supports recursive shape generation. To capture local details that are often used to refine shape descriptions, we build on top of vector-quantized deep implicit functions that generate a distribution of high-quality shapes. Results show that our method can generate shapes consistent with text descriptions, and shapes evolve gradually as more phrases are added. Our method supports shape editing, extrapolation, and can enable new applications in human--machine collaboration for creative design.
</details>

  [📄 Paper](https://arxiv.org/abs/2207.09446) | [🌐 Project Page](https://ivl.cs.brown.edu/#/projects/shapecrafter) 


### 5. Magic3D: High-Resolution Text-to-3D Content Creation [CVPR 2023]

**Authors**: Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin

<details span>
<summary><b>Abstract</b></summary>
 DreamFusion has recently demonstrated the utility of a pre-trained text-to-image diffusion model to optimize Neural Radiance Fields (NeRF), achieving remarkable text-to-3D synthesis results. However, the method has two inherent limitations: (a) extremely slow optimization of NeRF and (b) low-resolution image space supervision on NeRF, leading to low-quality 3D models with a long processing time. In this paper, we address these limitations by utilizing a two-stage optimization framework. First, we obtain a coarse model using a low-resolution diffusion prior and accelerate with a sparse 3D hash grid structure. Using the coarse representation as the initialization, we further optimize a textured 3D mesh model with an efficient differentiable renderer interacting with a high-resolution latent diffusion model. Our method, dubbed Magic3D, can create high quality 3D mesh models in 40 minutes, which is 2x faster than DreamFusion (reportedly taking 1.5 hours on average), while also achieving higher resolution. User studies show 61.7% raters to prefer our approach over DreamFusion. Together with the image-conditioned generation capabilities, we provide users with new ways to control 3D synthesis, opening up new avenues to various creative applications.
</details>

  [📄 Paper](https://arxiv.org/pdf/2211.10440.pdf) | [🌐 Project Page](https://research.nvidia.com/labs/dir/magic3d)  | [💻 Code][Coming soon.] 


### 6. Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models [CVPR 2023]

**Authors**: Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, Shenghua Gao

<details span>
<summary><b>Abstract</b></summary>
 Recent CLIP-guided 3D optimization methods, such as DreamFields and PureCLIPNeRF, have achieved impressive results in zero-shot text-to-3D synthesis. However, due to scratch training and random initialization without prior knowledge, these methods often fail to generate accurate and faithful 3D structures that conform to the input text. In this paper, we make the first attempt to introduce explicit 3D shape priors into the CLIP-guided 3D optimization process. Specifically, we first generate a high-quality 3D shape from the input text in the text-to-shape stage as a 3D shape prior. We then use it as the initialization of a neural radiance field and optimize it with the full prompt. To address the challenging text-to-shape generation task, we present a simple yet effective approach that directly bridges the text and image modalities with a powerful text-to-image diffusion model. To narrow the style domain gap between the images synthesized by the text-to-image diffusion model and shape renderings used to train the image-to-shape generator, we further propose to jointly optimize a learnable text prompt and fine-tune the text-to-image diffusion model for rendering-style image generation. Our method, Dream3D, is capable of generating imaginative 3D content with superior visual quality and shape accuracy compared to state-of-the-art methods.
</details>

  [📄 Paper](https://arxiv.org/pdf/2212.14704.pdf) | [🌐 Project Page](https://bluestyle97.github.io/dream3d/) | [💻 Code][Coming soon.] 


### 7. CLIP-Mesh: Generating textured meshes from text using pretrained image-text models [SIGGRAPH ASIA 2022]

**Authors**: Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, Tiberiu Popa

<details span>
<summary><b>Abstract</b></summary>
 We present a technique for zero-shot generation of a 3D model using only a target text prompt. Without any 3D supervision our method deforms the control shape of a limit subdivided surface along with its texture map and normal map to obtain a 3D asset that corresponds to the input text prompt and can be easily deployed into games or modeling applications. We rely only on a pre-trained CLIP model that compares the input text prompt with differentiably rendered images of our 3D model. While previous works have focused on stylization or required training of generative models we perform optimization on mesh parameters directly to generate shape, texture or both. To constrain the optimization to produce plausible meshes and textures we introduce a number of techniques using image augmentations and the use of a pretrained prior that generates CLIP image embeddings given a text embedding.
</details>

  [📄 Paper](https://arxiv.org/pdf/2203.13333.pdf) | [🌐 Project Page](https://www.nasir.lol/clipmesh) | [💻 Code](https://github.com/NasirKhalid24/CLIP-Mesh) 


### 8. Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation [CVPR 2023]

**Authors**: Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, Greg Shakhnarovich

<details span>
<summary><b>Abstract</b></summary>
 A diffusion model learns to predict a vector field of gradients. We propose to apply chain rule on the learned gradients, and back-propagate the score of a diffusion model through the Jacobian of a differentiable renderer, which we instantiate to be a voxel radiance field. This setup aggregates 2D scores at multiple camera viewpoints into a 3D score, and repurposes a pretrained 2D model for 3D data generation. We identify a technical challenge of distribution mismatch that arises in this application, and propose a novel estimation mechanism to resolve it. We run our algorithm on several off-the-shelf diffusion image generative models, including the recently released Stable Diffusion trained on the large-scale LAION dataset.
</details>

  [📄 Paper](https://arxiv.org/pdf/2212.00774.pdf) | [🌐 Project Page]() | [💻 Code](https://github.com/pals-ttic/sjc/) 


### 9. Dream Fields: Zero-Shot Text-Guided Object Generation with Dream Fields [CVPR 2022 and AI4CC 2022 (Best Poster)]

**Authors**: Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, Ben Poole

<details span>
<summary><b>Abstract</b></summary>
 We combine neural rendering with multi-modal image and text representations to synthesize diverse 3D objects solely from natural language descriptions. Our method, Dream Fields, can generate the geometry and color of a wide range of objects without 3D supervision. Due to the scarcity of diverse, captioned 3D data, prior methods only generate objects from a handful of categories, such as ShapeNet. Instead, we guide generation with image-text models pre-trained on large datasets of captioned images from the web. Our method optimizes a Neural Radiance Field from many camera views so that rendered images score highly with a target caption according to a pre-trained CLIP model. To improve fidelity and visual quality, we introduce simple geometric priors, including sparsity-inducing transmittance regularization, scene bounds, and new MLP architectures. In experiments, Dream Fields produce realistic, multi-view consistent object geometry and color from a variety of natural language captions.
</details>

  [📄 Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jain_Zero-Shot_Text-Guided_Object_Generation_With_Dream_Fields_CVPR_2022_paper.pdf) | [🌐 Project Page](https://ajayj.com/dreamfields) | [💻 Code]() 

### 10. RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D [arxiv 2023.11]

**Authors**: Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, Xiaoguang Han

<details span>
<summary><b>Abstract</b></summary>
 Lifting 2D diffusion for 3D generation is a challenging problem due to the lack of geometric prior and the com- plex entanglement of materials and lighting in natural im- ages. Existing methods have shown promise by first creat- ing the geometry through score-distillation sampling (SDS) applied to rendered surface normals, followed by appear- ance modeling. However, relying on a 2D RGB diffusion model to optimize surface normals is suboptimal due to the distribution discrepancy between natural images and nor- mals maps, leading to instability in optimization. In this paper, recognizing that the normal and depth information effectively describe scene geometry and be automatically estimated from images, we propose to learn a generaliz- able Normal-Depth diffusion model for 3D generation. We achieve this by training on the large-scale LAION dataset together with the generalizable image-to-depth and normal prior models. In an attempt to alleviate the mixed illumi- nation effects in the generated materials, we introduce an albedo diffusion model to impose data-driven constraints on the albedo component. Our experiments show that when in- tegrated into existing text-to-3D pipelines, our models sig- nificantly enhance the detail richness, achieving state-of- the-art results.
</details>

  [📄 Paper](https://arxiv.org/abs/2311.16918) | [🌐 Project Page](https://aigc3d.github.io/richdreamer/) | [💻 Code](https://github.com/modelscope/richdreamer) 



<br>

## Image to 3D Generation:
### 1. Zero-1-to-3: Zero-shot One Image to 3D Object [ICCV 2023]

**Authors**: Ruoshi Liu1, Rundi Wu1, Basile Van Hoorick1, Pavel Tokmakov2, Sergey Zakharov2, Carl Vondrick1

<details span>
<summary><b>Abstract</b></summary>
We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.
</details>

  [📄 Paper](https://arxiv.org/abs/2303.11328) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/cvlab-columbia/zero123) | [🤗 Hugging Face](https://huggingface.co/spaces/cvlab/zero123-live)

### 2. Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model [arxiv 2023.10]

**Authors**: Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, Hao Su

<details span>
<summary><b>Abstract</b></summary>
We report Zero123++, an image-conditioned diffusion model for generating 3D-consistent multi-view images from a single input view. To take full advantage of pretrained 2D generative priors, we develop various conditioning and training schemes to minimize the effort of finetuning from off-the-shelf image diffusion models such as Stable Diffusion. Zero123++ excels in producing high-quality, consistent multi-view images from a single image, overcoming common issues like texture degradation and geometric misalignment. Furthermore, we showcase the feasibility of training a ControlNet on Zero123++ for enhanced control over the generation process. The code is available at this https URL.
</details>

  [📄 Paper](https://arxiv.org/abs/2310.15110) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/SUDO-AI-3D/zero123plus) | [🤗 Hugging Face](https://huggingface.co/spaces/sudo-ai/zero123plus-demo-space)


### 3. One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization [arxiv 2023.06]

**Authors**: Minghua Liu1*, Chao Xu2*, Haian Jin3,4*, Linghao Chen1,4*, Mukund Varma T5, Zexiang Xu6, Hao Su1

<details span>
<summary><b>Abstract</b></summary>
 Single image 3D reconstruction is an important but challenging task that requires extensive knowledge of our natural world. Many existing methods solve this problem by optimizing a neural radiance field under the guidance of 2D diffusion models but suffer from lengthy optimization time, 3D inconsistency results, and poor geometry. In this work, we propose a novel method that takes a single image of any object as input and generates a full 360-degree 3D textured mesh in a single feed-forward pass. Given a single image, we first use a view-conditioned 2D diffusion model, Zero123, to generate multi-view images for the input view, and then aim to lift them up to 3D space. Since traditional reconstruction methods struggle with inconsistent multi-view predictions, we build our 3D reconstruction module upon an SDF-based generalizable neural surface reconstruction method and propose several critical training strategies to enable the reconstruction of 360-degree meshes. Without costly optimizations, our method reconstructs 3D shapes in significantly less time than existing methods. Moreover, our method favors better geometry, generates more 3D consistent results, and adheres more closely to the input image. We evaluate our approach on both synthetic data and in-the-wild images and demonstrate its superiority in terms of both mesh quality and runtime. In addition, our approach can seamlessly support the text-to-3D task by integrating with off-the-shelf text-to-image diffusion models.
</details>

  [📄 Paper](https://arxiv.org/pdf/2306.16928.pdf) | [🌐 Project Page](https://one-2-3-45.github.io/) | [💻 Code](https://github.com/One-2-3-45/One-2-3-45) 


### 4. One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion [arxiv 2023.11]

**Authors**: Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, Hao Su

<details span>
<summary><b>Abstract</b></summary>
Recent advancements in open-world 3D object generation have been remarkable, with image-to-3D methods offering superior fine-grained control over their text-to-3D counterparts. However, most existing models fall short in simultaneously providing rapid generation speeds and high fidelity to input images - two features essential for practical applications. In this paper, we present One-2-3-45++, an innovative method that transforms a single image into a detailed 3D textured mesh in approximately one minute. Our approach aims to fully harness the extensive knowledge embedded in 2D diffusion models and priors from valuable yet limited 3D data. This is achieved by initially finetuning a 2D diffusion model for consistent multi-view image generation, followed by elevating these images to 3D with the aid of multi-view conditioned 3D native diffusion models. Extensive experimental evaluations demonstrate that our method can produce high-quality, diverse 3D assets that closely mirror the original input image. Our project webpage: this https URL.
</details>

  [📄 Paper](https://arxiv.org/pdf/2311.07885.pdf) | [🌐 Project Page](https://sudo-ai-3d.github.io/One2345plus_page/) | [💻 Code](https://github.com/SUDO-AI-3D/One2345plus) 


### 5. Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model [ICLR 2024 under review]

**Authors**: 

<details span>
<summary><b>Abstract</b></summary>
 Text-to-3D with diffusion models has achieved remarkable progress in recent years. However, existing methods either rely on score distillation-based optimization which suffer from slow inference, low diversity and Janus problems, or are feed-forward methods that generate low quality results due to the scarcity of 3D training data. In this paper, we propose Instant3D, a novel method that generates high-quality and diverse 3D assets from text prompts in a feed-forward manner. We adopt a two-stage paradigm, which first generates a sparse set of four structured and consistent views from text in one shot with a fine-tuned 2D text-to-image diffusion model, and then directly regresses the NeRF from the generated images with a novel transformer-based sparse-view reconstructor. Through extensive experiments, we demonstrate that our method can generate high-quality, diverse and Janus-free 3D assets within 20 seconds, which is two order of magnitude faster than previous optimization-based methods that can take 1 to 10 hours.
</details>

  [📄 Paper](https://openreview.net/forum?id=2lDQLiH1W4) | [🌐 Project Page](https://instant-3d.github.io/) 


### 6. Wonder3d: Single Image to 3D using Cross-Domain Diffusion [arxiv 2023.10]

**Authors**: Xiaoxiao Long1,3,6,*, Yuan-Chen Guo2,3,*, Cheng Lin1, Yuan Liu1, Zhiyang Dou1, Lingjie Liu4, Yuexin Ma5, Song-Hai Zhang2, Marc Habermann6, Christian Theobalt6, Wenping Wang7

<details span>
<summary><b>Abstract</b></summary>
 In this work, we introduce Wonder3D, a novel method for efficiently generating high-fidelity textured meshes from single-view images.Recent methods based on Score Distillation Sampling (SDS) have shown the potential to recover 3D geometry from 2D diffusion priors, but they typically suffer from time-consuming per-shape optimization and inconsistent geometry. In contrast, certain works directly produce 3D information via fast network inferences, but their results are often of low quality and lack geometric details. To holistically improve the quality, consistency, and efficiency of image-to-3D tasks, we propose a cross-domain diffusion model that generates multi-view normal maps and the corresponding color images. To ensure consistency, we employ a multi-view cross-domain attention mechanism that facilitates information exchange across views and modalities. Lastly, we introduce a geometry-aware normal fusion algorithm that extracts high-quality surfaces from the multi-view 2D representations. Our extensive evaluations demonstrate that our method achieves high-quality reconstruction results, robust generalization, and reasonably good efficiency compared to prior works.
</details>

  [📄 Paper](https://arxiv.org/pdf/2310.15008.pdf) | [🌐 Project Page](https://www.xxlong.site/Wonder3D/) | [💻 Code](https://github.com/xxlong0/Wonder3D) 


### 7. LRM: Large Reconstruction Model for Single Image to 3D [ICLR 2024]

**Authors**: Yicong Hong1,2, Kai Zhang1, Jiuxiang Gu1, Sai Bi1, Yang Zhou1, Difan Liu1, Feng Liu1, Kalyan Sunkavalli1, Trung Bui1, Hao Tan1

<details span>
<summary><b>Abstract</b></summary>
 We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs including real-world in-the-wild captures and images from generative models.
</details>

  [📄 Paper](https://arxiv.org/abs/2311.04400) | [🌐 Project Page](https://yiconghong.me/LRM/)


### 8. Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors [arxiv 2023.06]

**Authors**: Qian, Guocheng and Mai, Jinjie and Hamdi, Abdullah and Ren, Jian and Siarohin, Aliaksandr and Li, Bing and Lee, Hsin-Ying and Skorokhodov, Ivan and Wonka, Peter and Tulyakov, Sergey and Ghanem, Bernard

<details span>
<summary><b>Abstract</b></summary>
 We present "Magic123", a two-stage coarse-to-fine solution for high-quality, textured 3D meshes generation from a single unposed image in the wild using both 2D and 3D priors. In the first stage, we optimize a neural radiance field to produce a coarse geometry. In the second stage, we adopt a memory-efficient differentiable mesh representation to yield a high-resolution mesh with a visually appealing texture. In both stages, the 3D content is learned through reference view supervision and novel views guided by both 2D and 3D diffusion priors. We introduce a single tradeoff parameter between the 2D and 3D priors to control exploration (more imaginative) and exploitation (more precise) of the generated geometry. Additionally, We employ textual inversion and monocular depth regularization to encourage consistent appearances across views and to prevent degenerate solutions, respectively. Magic123 demonstrates a significant improvement over previous image-to-3D techniques, as validated through extensive experiments on synthetic benchmarks and diverse real-world images.
</details>

  [📄 Paper](https://arxiv.org/abs/2306.17843) | [🌐 Project Page](https://guochengqian.github.io/project/magic123/) | [💻 Code](https://github.com/guochengqian/Magic123) 


### 9. DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior [arxiv 2023.10]

**Authors**: Jingxiang Sun 1,   Bo Zhang 3,   Ruizhi Shao 1,   Lizhen Wang 1,   Wen Liu 2,   Zhenda Xie 2,   Yebin Liu 1

<details span>
<summary><b>Abstract</b></summary>
 We present DreamCraft3D, a hierarchical 3D content generation method that produces high-fidelity and coherent 3D objects. We tackle the problem by leveraging a 2D reference image to guide the stages of geometry sculpting and texture boosting. A central focus of this work is to address the consistency issue that existing works encounter. To sculpt geometries that render coherently, we perform score distillation sampling via a view-dependent diffusion model. This 3D prior, alongside several training strategies, prioritizes the geometry consistency but compromises the texture fidelity. We further propose Bootstrapped Score Distillation (BSD) to specifically boost the texture. We train a personalized diffusion model, Dreambooth, on the augmented renderings of the scene, imbuing it with 3D knowledge of the scene being optimized. The score distillation from this 3D-aware diffusion prior provides view-consistent guidance for the scene. Notably, through an alternating optimization of the diffusion prior and 3D scene representation, we achieve mutually reinforcing improvements: the optimized 3D scene aids in training the scene-specific diffusion model, which offers increasingly view-consistent guidance for 3D optimization. The optimization is thus bootstrapped and leads to substantial texture boosting. With tailored 3D priors throughout the hierarchical generation, DreamCraft3D generates coherent 3D objects with photorealistic renderings, advancing the state-of-the-art in 3D content generation.
</details>

  [📄 Paper](https://arxiv.org/abs/2310.16818) | [🌐 Project Page](https://mrtornado24.github.io/DreamCraft3D/) | [💻 Code](https://github.com/deepseek-ai/DreamCraft3D) 



<br>

## Audio to 3D Generation:

### 1. From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations [arxiv 2024.01]


**Authors**: Evonne Ng1, 2, Javier Romero1, Timur Bagautdinov1, Shaojie Bai1, Trevor Darrell2, Angjoo Kanazawa2, Alexander Richard1

<details span>
<summary><b>Abstract</b></summary>
We present a framework for generating full-bodied photorealistic avatars that gesture according to the conversational dynamics of a dyadic interaction. Given speech audio, we output multiple possibilities of gestural motion for an individual, including face, body, and hands. The key behind our method is in combining the benefits of sample diversity from vector quantization with the high-frequency details obtained through diffusion to generate more dynamic, expressive motion. We visualize the generated motion using highly photorealistic avatars that can express crucial nuances in gestures (e.g. sneers and smirks). To facilitate this line of research, we introduce a first-of-its-kind multi-view conversational dataset that allows for photorealistic reconstruction. Experiments show our model generates appropriate and diverse gestures, outperforming both diffusion- and VQ-only methods. Furthermore, our perceptual evaluation highlights the importance of photorealism (vs. meshes) in accurately assessing subtle motion details in conversational gestures. Code and dataset will be publicly released.
</details>

  [📄 Paper](https://people.eecs.berkeley.edu/~evonne_ng/projects/audio2photoreal/static/CCA.pdf) | [🌐 Project Page](https://people.eecs.berkeley.edu/~evonne_ng/projects/audio2photoreal/) | [💻 Code]([https://github.com/deepseek-ai/DreamCraft3D](https://github.com/facebookresearch/audio2photoreal)) 



<be>

## Editing:
### 1. DreamEditor: Text-Driven 3D Scene Editing with Neural Fields [arxiv 2023.06]

**Authors**: Zhuang, Jingyu and Wang, Chen and Liu, Lingjie and Lin, Liang and Li, Guanbin

<details span>
<summary><b>Abstract</b></summary>
Neural fields have achieved impressive advancements in view synthesis and scene reconstruction. However, editing these neural fields remains challenging due to the implicit encoding of geometry and texture information. In this paper, we propose DreamEditor, a novel framework that enables users to perform controlled editing of neural fields using text prompts. By representing scenes as mesh-based neural fields, DreamEditor allows localized editing within specific regions. DreamEditor utilizes the text encoder of a pretrained text-to-Image diffusion model to automatically identify the regions to be edited based on the semantics of the text prompts. Subsequently, DreamEditor optimizes the editing region and aligns its geometry and texture with the text prompts through score distillation sampling [Poole et al. 2022]. Extensive experiments have demonstrated that DreamEditor can accurately edit neural fields of real-world scenes according to the given text prompts while ensuring consistency in irrelevant areas. DreamEditor generates highly realistic textures and geometry, significantly surpassing previous works in both quantitative and qualitative evaluations.
</details>

  [📄 Paper](https://arxiv.org/abs/2306.13455) | [🌐 Project Page](https://www.sysu-hcp.net/projects/cv/111.html) | [💻 Code](https://github.com/zjy526223908/DreamEditor) 


### 2. IDE-3D: Interactive Disentangled Editing For High-Resolution 3D-aware Portrait Synthesis [arxiv 2022.05]

**Authors**: Jingxiang Sun, Xuan Wang, Yichun Shi, Lizhen Wang, Jue Wang, Yebin Liu

<details span>
<summary><b>Abstract</b></summary>
Existing 3D-aware facial generation methods face a dilemma in quality versus editability: they either generate editable results in low resolution, or high quality ones with no editing flexibility. In this work, we propose a new approach that brings the best of both worlds together. Our system consists of three major components: (1) a 3D-semantics-aware generative model that produces view-consistent, disentangled face images and semantic masks; (2) a hybrid GAN inversion approach that initialize the latent codes from the semantic and texture encoder, and further optimized them for faithful reconstruction; and (3) a canonical editor that enables efficient manipulation of semantic masks in canonical view and producs high quality editing results. Our approach is competent for many applications, e.g. free-view face drawing, editing and style control. Both quantitative and qualitative results show that our method reaches the state-of-the-art in terms of photorealism, faithfulness and efficiency.
</details>

  [📄 Paper](https://arxiv.org/abs/2205.15517) | [🌐 Project Page](https://mrtornado24.github.io/IDE-3D/) | [💻 Code](https://github.com/MrTornado24/IDE-3D) 



### 3. DM-NeRF: 3D Scene Geometry Decomposition and Manipulation from 2D Images [ICLR 2023]

**Authors**: Bing Wang, Lu Chen, Bo Yang

<details span>
<summary><b>Abstract</b></summary>
 In this paper, we study the problem of 3D scene geometry decomposition and manipulation from 2D views. By leveraging the recent implicit neural representation techniques, particularly the appealing neural radiance fields, we introduce an object field component to learn unique codes for all individual objects in 3D space only from 2D supervision. The key to this component is a series of carefully designed loss functions to enable every 3D point, especially in non-occupied space, to be effectively optimized even without 3D labels. In addition, we introduce an inverse query algorithm to freely manipulate any specified 3D object shape in the learned scene representation. Notably, our manipulation algorithm can explicitly tackle key issues such as object collisions and visual occlusions. Our method, called DM-NeRF, is among the first to simultaneously reconstruct, decompose, manipulate and render complex 3D scenes in a single pipeline. Extensive experiments on three datasets clearly show that our method can accurately decompose all 3D objects from 2D views, allowing any interested object to be freely manipulated in 3D space such as translation, rotation, size adjustment, and deformation.
</details>

  [📄 Paper](https://arxiv.org/abs/2208.07227) | [💻 Code](https://github.com/vLAR-group/DM-NeRF) 


### 4. Image Sculpting: Precise Object Editing with 3D Geometry Control [arxiv 2024.01]

**Authors**: Jiraphon Yenphraphai, Xichen Pan, Sainan Liu, Daniele Panozzo, Saining Xie

<details span>
<summary><b>Abstract</b></summary>
 We present Image Sculpting, a new framework for editing 2D images by incorporating tools from 3D geometry and graphics. This approach differs markedly from existing methods, which are confined to 2D spaces and typically rely on textual instructions, leading to ambiguity and limited control. Image Sculpting converts 2D objects into 3D, enabling direct interaction with their 3D geometry. Post-editing, these objects are re-rendered into 2D, merging into the original image to produce high-fidelity results through a coarse-to-fine enhancement process. The framework supports precise, quantifiable, and physically-plausible editing options such as pose editing, rotation, translation, 3D composition, carving, and serial addition. It marks an initial step towards combining the creative freedom of generative models with the precision of graphics pipelines.
</details>

  [📄 Paper](https://arxiv.org/abs/2401.01702) | [🌐 Project Page](https://image-sculpting.github.io/) | [💻 Code](https://github.com/vision-x-nyu/image-sculpting) 


<br>

## Human-Avatar Generation:
### 1. AvatarBooth: High-Quality and Customizable 3D Human Avatar Generation [arxiv 2023.06]

**Authors**: Yifei Zeng1, Yuanxun Lu1, Xinya Ji1, Yao Yao1, Hao Zhu1, Xun Cao1,

<details span>
<summary><b>Abstract</b></summary>
 We introduce AvatarBooth, a novel method for generating high-quality 3D avatars using text prompts or specific images. Unlike previous approaches that can only synthesize avatars based on simple text descriptions, our method enables the creation of personalized avatars from casually captured face or body images, while still supporting text-based model generation and editing. Our key contribution is the precise avatar generation control by using dual fine-tuned diffusion models separately for the human face and body. This enables us to capture intricate details of facial appearance, clothing, and accessories, resulting in highly realistic avatar generations. Furthermore, we introduce pose-consistent constraint to the optimization process to enhance the multi-view consistency of synthesized head images from the diffusion model and thus eliminate interference from uncontrolled human poses. In addition, we present a multi-resolution rendering strategy that facilitates coarse-to-fine supervision of 3D avatar generation, thereby enhancing the performance of the proposed system. The resulting avatar model can be further edited using additional text descriptions and driven by motion sequences. Experiments show that AvatarBooth outperforms previous text-to-3D methods in terms of rendering and geometric quality from either text prompts or specific images. Please check our project website at this https URL.
</details>

  [📄 Paper](https://arxiv.org/pdf/2306.09864) | [🌐 Project Page](https://zeng-yifei.github.io/avatarbooth_page/) | [💻 Code](https://github.com/zeng-yifei/AvatarBooth) 



### 2. SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance [arxiv 2023.12]

**Authors**: Yuanyou Xu, Zongxin Yang, Yi Yang

<details span>
<summary><b>Abstract</b></summary>
 Powered by large-scale text-to-image generation models, text-to-3D avatar generation has made promising progress. However, most methods fail to produce photorealistic results, limited by imprecise geometry and low-quality appearance. Towards more practical avatar generation, we present SEEAvatar, a method for generating photorealistic 3D avatars from text with SElf-Evolving constraints for decoupled geometry and appearance. For geometry, we propose to constrain the optimized avatar in a decent global shape with a template avatar. The template avatar is initialized with human prior and can be updated by the optimized avatar periodically as an evolving template, which enables more flexible shape generation. Besides, the geometry is also constrained by the static human prior in local parts like face and hands to maintain the delicate structures. For appearance generation, we use diffusion model enhanced by prompt engineering to guide a physically based rendering pipeline to generate realistic textures. The lightness constraint is applied on the albedo texture to suppress incorrect lighting effect. Experiments show that our method outperforms previous methods on both global and local geometry and appearance quality by a large margin. Since our method can produce high-quality meshes and textures, such assets can be directly applied in classic graphics pipeline for realistic rendering under any lighting condition. Project page at: this https URL.
</details>

  [📄 Paper](https://arxiv.org/pdf/2312.08889.pdf) | [🌐 Project Page](https://seeavatar3d.github.io/) 



### 3. Text and Image Guided 3D Avatar Generation and Manipulation [arxiv 2023.08]

**Authors**: Zehranaz Canfes, M. Furkan Atasoy, Alara Dirik, Pinar Yanardag

<details span>
<summary><b>Abstract</b></summary>
 The manipulation of latent space has recently become an interesting topic in the field of generative models. Recent research shows that latent directions can be used to manipulate images towards certain attributes. However, controlling the generation process of 3D generative models remains a challenge. In this work, we propose a novel 3D manipulation method that can manipulate both the shape and texture of the model using text or image-based prompts such as 'a young face' or 'a surprised face'. We leverage the power of Contrastive Language-Image Pre-training (CLIP) model and a pre-trained 3D GAN model designed to generate face avatars, and create a fully differentiable rendering pipeline to manipulate meshes. More specifically, our method takes an input latent code and modifies it such that the target attribute specified by a text or image prompt is present or enhanced, while leaving other attributes largely unaffected. Our method requires only 5 minutes per manipulation, and we demonstrate the effectiveness of our approach with extensive results and comparisons.
</details>

  [📄 Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Canfes_Text_and_Image_Guided_3D_Avatar_Generation_and_Manipulation_WACV_2023_paper.pdf) | [🌐 Project Page](https://catlab-team.github.io/latent3D/) | [💻 Code](https://github.com/catlab-team/latent3D_code) 


### 4. En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data [arxiv 2024.01]

**Authors**: Yifang Men1, Biwen Lei1, Yuan Yao1, Miaomiao Cui1, Zhouhui Lian2, Xuansong Xie1

<details span>
<summary><b>Abstract</b></summary>
 We present En3D, an enhanced generative scheme for sculpting high-quality 3D human avatars. Unlike previous works that rely on scarce 3D datasets or limited 2D collections with imbalanced viewing angles and imprecise pose priors, our approach aims to develop a zero-shot 3D generative scheme capable of producing visually realistic, geometrically accurate and content-wise diverse 3D humans without relying on pre-existing 3D or 2D assets. To address this challenge, we introduce a meticulously crafted workflow that implements accurate physical modeling to learn the enhanced 3D generative model from synthetic 2D data. During inference, we integrate optimization modules to bridge the gap between realistic appearances and coarse 3D shapes. Specifically, En3D comprises three modules: a 3D generator that accurately models generalizable 3D humans with realistic appearance from synthesized balanced, diverse, and structured human images; a geometry sculptor that enhances shape quality using multi-view normal constraints for intricate human anatomy; and a texturing module that disentangles explicit texture maps with fidelity and editability, leveraging semantical UV partitioning and a differentiable rasterizer. Experimental results show that our approach significantly outperforms prior works in terms of image quality, geometry accuracy and content diversity. We also showcase the applicability of our generated avatars for animation and editing, as well as the scalability of our approach for content-style free adaptation.
</details>

  [📄 Paper](http://arxiv.org/abs/2401.01173) | [🌐 Project Page](https://menyifang.github.io/projects/En3D/index.html) | [💻 Code](https://github.com/menyifang/En3D) 


<br>

## Autonomous Driving:

### 1. 3D Object Detection for Autonomous Driving: A Comprehensive Survey [IJCV 2023]

**Authors**: Jiageng Mao, Shaoshuai Shi, Xiaogang Wang, Hongsheng Li

<details span>
<summary><b>Abstract</b></summary>
 Autonomous driving, in recent years, has been receiving increasing attention for its potential to relieve drivers' burdens and improve the safety of driving. In modern autonomous driving pipelines, the perception system is an indispensable component, aiming to accurately estimate the status of surrounding environments and provide reliable observations for prediction and planning. 3D object detection, which intelligently predicts the locations, sizes, and categories of the critical 3D objects near an autonomous vehicle, is an important part of a perception system. This paper reviews the advances in 3D object detection for autonomous driving. First, we introduce the background of 3D object detection and discuss the challenges in this task. Second, we conduct a comprehensive survey of the progress in 3D object detection from the aspects of models and sensory inputs, including LiDAR-based, camera-based, and multi-modal detection approaches. We also provide an in-depth analysis of the potentials and challenges in each category of methods. Additionally, we systematically investigate the applications of 3D object detection in driving systems. Finally, we conduct a performance analysis of the 3D object detection approaches, and we further summarize the research trends over the years and prospect the future directions of this area.
</details>

  [📄 Paper]() | [💻 Code](https://github.com/PointsCoder/Awesome-3D-Object-Detection-for-Autonomous-Driving) 


### 2. A Survey on Safety-Critical Driving Scenario Generation – A Methodological Perspective [IEEE Transactions on Intelligent Transportation Systems (T-ITS) 2023]

**Authors**: Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li, Ding Zhao

<details span>
<summary><b>Abstract</b></summary>
 Autonomous driving systems have witnessed a significant development during the past years thanks to the advance in machine learning-enabled sensing and decision-making algorithms. One critical challenge for their massive deployment in the real world is their safety evaluation. Most existing driving systems are still trained and evaluated on naturalistic scenarios collected from daily life or heuristically-generated adversarial ones. However, the large population of cars, in general, leads to an extremely low collision rate, indicating that the safety-critical scenarios are rare in the collected real-world data. Thus, methods to artificially generate scenarios become crucial to measure the risk and reduce the cost. In this survey, we focus on the algorithms of safety-critical scenario generation in autonomous driving. We first provide a comprehensive taxonomy of existing algorithms by dividing them into three categories: data-driven generation, adversarial generation, and knowledge-based generation. Then, we discuss useful tools for scenario generation, including simulation platforms and packages. Finally, we extend our discussion to five main challenges of current works -- fidelity, efficiency, diversity, transferability, controllability -- and research opportunities lighted up by these challenges.
</details>

  [📄 Paper](https://arxiv.org/abs/2202.02215) 



### 3. Street Gaussians for Modeling Dynamic Urban Scenes [arxiv 2024.01]

**Authors**: Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, Sida Peng

<details span>
<summary><b>Abstract</b></summary>
 This paper aims to tackle the problem of modeling dynamic urban street scenes from monocular videos. Recent methods extend NeRF by incorporating tracked vehicle poses to animate vehicles, enabling photo-realistic view synthesis of dynamic urban street scenes. However, significant limitations are their slow training and rendering speed, coupled with the critical need for high precision in tracked vehicle poses. We introduce Street Gaussians, a new explicit scene representation that tackles all these limitations. Specifically, the dynamic urban street is represented as a set of point clouds equipped with semantic logits and 3D Gaussians, each associated with either a foreground vehicle or the background. To model the dynamics of foreground object vehicles, each object point cloud is optimized with optimizable tracked poses, along with a dynamic spherical harmonics model for the dynamic appearance. The explicit representation allows easy composition of object vehicles and background, which in turn allows for scene editing operations and rendering at 133 FPS (1066×1600 resolution) within half an hour of training. The proposed method is evaluated on multiple challenging benchmarks, including KITTI and Waymo Open datasets. Experiments show that the proposed method consistently outperforms state-of-the-art methods across all datasets. Furthermore, the proposed representation delivers performance on par with that achieved using precise ground-truth poses, despite relying only on poses from an off-the-shelf tracker. The code is available at this https URL.
</details>

  [📄 Paper](https://arxiv.org/abs/2401.01339) | [🌐 Project Page](https://zju3dv.github.io/street_gaussians/) | [💻 Code](https://github.com/zju3dv/street_gaussians) 





<br>

## BioMedical:

### 1. MinD-3D: Reconstruct High-quality 3D objects in Human Brain [arxiv 2023.12]
**Authors**: Jianxiong Gao, Yuqian Fu, Yun Wang, Xuelin Qian, Jianfeng Feng, Yanwei Fu

<details span>
<summary><b>Abstract</b></summary>
In this paper, we introduce Recon3DMind, a groundbreaking task focused on reconstructing 3D visuals from Functional Magnetic Resonance Imaging (fMRI) signals. This represents a major step forward in cognitive neuroscience and computer vision. To support this task, we present the fMRI-Shape dataset, utilizing 360-degree view videos of 3D objects for comprehensive fMRI signal capture. Containing 55 categories of common objects from daily life, this dataset will bolster future research endeavors. We also propose MinD-3D, a novel and effective three-stage framework that decodes and reconstructs the brain's 3D visual information from fMRI signals. This method starts by extracting and aggregating features from fMRI frames using a neuro-fusion encoder, then employs a feature bridge diffusion model to generate corresponding visual features, and ultimately recovers the 3D object through a generative transformer decoder. Our experiments demonstrate that this method effectively extracts features that are valid and highly correlated with visual regions of interest (ROIs) in fMRI signals. Notably, it not only reconstructs 3D objects with high semantic relevance and spatial similarity but also significantly deepens our understanding of the human brain's 3D visual processing capabilities. Project page at:  https://jianxgao.github.io/MinD-3D.
</details>

  [📄 Paper](https://arxiv.org/abs/2312.07485) | [🌐 Project Page](https://jianxgao.github.io/MinD-3D/) | [💻 Code](https://github.com/JianxGao/MinD-3D) 


<br>

## 4D AIGC


### 1. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency [arxiv 2023.12]

**Authors**: Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, Yunchao Wei

<details span>
<summary><b>Abstract</b></summary>
 Aided by text-to-image and text-to-video diffusion models, existing 4D content creation pipelines utilize score distillation sampling to optimize the entire dynamic 3D scene. However, as these pipelines generate 4D content from text or image inputs, they incur significant time and effort in prompt engineering through trial and error. This work introduces 4DGen, a novel, holistic framework for grounded 4D content creation that decomposes the 4D generation task into multiple stages. We identify static 3D assets and monocular video sequences as key components in constructing the 4D content. Our pipeline facilitates conditional 4D generation, enabling users to specify geometry (3D assets) and motion (monocular videos), thus offering superior control over content creation. Furthermore, we construct our 4D representation using dynamic 3D Gaussians, which permits efficient, high-resolution supervision through rendering during training, thereby facilitating high-quality 4D generation. Additionally, we employ spatial-temporal pseudo labels on anchor frames, along with seamless consistency priors implemented through 3D-aware score distillation sampling and smoothness regularizations. Compared to existing baselines, our approach yields competitive results in faithfully reconstructing input signals and realistically inferring renderings from novel viewpoints and timesteps. Most importantly, our method supports grounded generation, offering users enhanced control, a feature difficult to achieve with previous methods. Project page: this https URL
</details>

  [📄 Paper](https://arxiv.org/abs/2312.17225) | [🌐 Project Page](https://vita-group.github.io/4DGen/) | [💻 Code](https://github.com/VITA-Group/4DGen) 

### 2. DreamGaussian4D: Generative 4D Gaussian Splatting [arxiv 2023.12]

**Authors**: Jiawei Ren* Liang Pan* Jiaxiang Tang Chi Zhang Ang Cao Gang Zeng Ziwei Liu†

<details span>
<summary><b>Abstract</b></summary>
 Remarkable progress has been made in 4D content generation recently. However, existing methods suffer from long optimization time, lack of motion controllability, and a low level of detail. In this paper, we introduce DreamGaussian4D, an efficient 4D generation framework that builds on 4D Gaussian Splatting representation. Our key insight is that the explicit modeling of spatial transformations in Gaussian Splatting makes it more suitable for the 4D generation setting compared with implicit representations. DreamGaussian4D reduces the optimization time from several hours to just a few minutes, allows flexible control of the generated 3D motion, and produces animated meshes that can be efficiently rendered in 3D engines.
</details>

  [📄 Paper](https://arxiv.org/abs/2312.17142) | [🌐 Project Page](https://jiawei-ren.github.io/projects/dreamgaussian4d/) | [💻 Code](https://github.com/jiawei-ren/dreamgaussian4d) 


### 3. Learning 3D Animal Motions from Unlabeled Online Videos [arxiv 2023.12]

**Authors**: Keqiang Sun1*, Dor Litvak2,3*, Yunzhi Zhang2, Hongsheng Li1, Jiajun Wu2†, Shangzhe Wu2†

<details span>
<summary><b>Abstract</b></summary>
We introduce Ponymation, a new method for learning a generative model of articulated 3D animal motions from raw, unlabeled online videos. Unlike existing approaches for motion synthesis, our model does not require any pose annotations or parametric shape models for training, and is learned purely from a collection of raw video clips obtained from the Internet. We build upon a recent work, MagicPony, which learns articulated 3D animal shapes purely from single image collections, and extend it on two fronts. First, instead of training on static images, we augment the framework with a video training pipeline that incorporates temporal regularizations, achieving more accurate and temporally consistent reconstructions. Second, we learn a generative model of the underlying articulated 3D motion sequences via a spatio-temporal transformer VAE, simply using 2D reconstruction losses without relying on any explicit pose annotations. At inference time, given a single 2D image of a new animal instance, our model reconstructs an articulated, textured 3D mesh, and generates plausible 3D animations by sampling from the learned motion latent space.
</details>

  [📄 Paper](https://arxiv.org/pdf/2312.13604.pdf) | [🌐 Project Page](https://keqiangsun.github.io/projects/ponymation/) | [💻 Code](https://keqiangsun.github.io/projects/ponymationn) 






## Misc:

### 1. FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding [arxiv 2024.01]

**Authors**: Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, Mingyang Li

<details span>
<summary><b>Abstract</b></summary>
 Precisely perceiving the geometric and semantic properties of real-world 3D objects is crucial for the continued evolution of augmented reality and robotic applications. To this end, we present \algfull{} (\algname{}), which incorporates vision-language embeddings of foundation models into 3D Gaussian Splatting (GS). The key contribution of this work is an efficient method to reconstruct and represent 3D vision-language models. This is achieved by distilling feature maps generated from image-based foundation models into those rendered from our 3D model. To ensure high-quality rendering and fast training, we introduce a novel scene representation by integrating strengths from both GS and multi-resolution hash encodings (MHE). Our effective training procedure also introduces a pixel alignment loss that makes the rendered feature distance of same semantic entities close, following the pixel-level semantic boundaries. Our results demonstrate remarkable multi-view semantic consistency, facilitating diverse downstream tasks, beating state-of-the-art methods by 10.2 percent on open-vocabulary language-based object detection, despite that we are 851× faster for inference. This research explores the intersection of vision, language, and 3D scene representation, paving the way for enhanced scene understanding in uncontrolled real-world environments. We plan to release the code upon paper acceptance.
</details>

  [📄 Paper](https://arxiv.org/abs/2401.01970) 

### 2. Learning the 3D Fauna of the Web [arxiv 2401]

**Authors**: Zizhang Li, Dor Litvak, Ruining Li, Yunzhi Zhang, Tomas Jakab, Christian Rupprecht, Shangzhe Wu, Andrea Vedaldi, Jiajun Wu


<details span>
<summary><b>Abstract</b></summary>
 Learning 3D models of all animals on the Earth requires massively scaling up existing solutions. With this ultimate goal in mind, we develop 3D-Fauna, an approach that learns a pan-category deformable 3D animal model for more than 100 animal species jointly. One crucial bottleneck of modeling animals is the limited availability of training data, which we overcome by simply learning from 2D Internet images. We show that prior category-specific attempts fail to generalize to rare species with limited training images. We address this challenge by introducing the Semantic Bank of Skinned Models (SBSM), which automatically discovers a small set of base animal shapes by combining geometric inductive priors with semantic knowledge implicitly captured by an off-the-shelf self-supervised feature extractor. To train such a model, we also contribute a new large-scale dataset of diverse animal species. At inference time, given a single image of any quadruped animal, our model reconstructs an articulated 3D mesh in a feed-forward fashion within seconds.
</details>

  [📄 Paper](https://arxiv.org/abs/2401.02400) | [🌐 Project Page](https://kyleleey.github.io/3DFauna/) | [💻 Code]() 



### 3. M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts [arxiv 2023.12]

**Authors**: Mingsheng Li, Xin Chen, Chi Zhang, Sijin Chen, Hongyuan Zhu, Fukun Yin, Gang Yu, Tao Chen

<details span>
<summary><b>Abstract</b></summary>
 Recently, 3D understanding has become popular to facilitate autonomous agents to perform further decisionmaking. However, existing 3D datasets and methods are often limited to specific tasks. On the other hand, recent progress in Large Language Models (LLMs) and Multimodal Language Models (MLMs) have demonstrated exceptional general language and imagery tasking performance. Therefore, it is interesting to unlock MLM's potential to be 3D generalist for wider tasks. However, current MLMs' research has been less focused on 3D tasks due to a lack of large-scale 3D instruction-following datasets. In this work, we introduce a comprehensive 3D instructionfollowing dataset called M3DBench, which possesses the following characteristics: 1) It supports general multimodal instructions interleaved with text, images, 3D objects, and other visual prompts. 2) It unifies diverse 3D tasks at both region and scene levels, covering a variety of fundamental abilities in real-world 3D environments. 3) It is a large-scale 3D instruction-following dataset with over 320k instruction-response pairs. Furthermore, we establish a new benchmark for assessing the performance of large models in understanding multi-modal 3D prompts. Extensive experiments demonstrate the effectiveness of our dataset and baseline, supporting general 3D-centric tasks, which can inspire future research.
</details>

  [📄 Paper](https://arxiv.org/abs/2312.10763) | [🌐 Project Page](https://m3dbench.github.io/) | [💻 Code](https://github.com/OpenM3D/M3DBench) 


<!--

### 2.

**Authors**: 

<details span>
<summary><b>Abstract</b></summary>
</details>

  [📄 Paper]() | [🌐 Project Page]() | [💻 Code]() 


-->



<br>


## Open Source Implementations
### Unofficial Implementations
  [ThreeStudio](https://github.com/threestudio-project/threestudio) is a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.
  
Currenty supported 3D content generation methods (2023.12.29):

ProlificDreamer | DreamFusion | Magic3D | SJC | Latent-NeRF | Fantasia3D | TextMesh | Zero-1-to-3 | Magic123 | InstructNeRF2NeRF | Control4D 



## Credits

- Thanks to [MrNeRF](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) for inspiring me to construct this repo.
