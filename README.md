# Awesome 3D AIGC Resources

A curated list of papers and open-source resources focused on 3D AIGC, intended to keep pace with the anticipated surge of research in the coming months. If you have any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.


## Table of contents

- [Survey](#survey)
- [Text to 3D Generation](#text-to-3d-generation)
- [Image to 3D Generation](#image-to-3d-generation)
- [3D Editing](#editing)
- [Human Avatar Generation](#human-avatar-generation)
- [Autonomous Driving](#autonomous-driving)
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
<br>

  **Dec 29, 2023**: Contribute to the section on text-to-3d by adding new papers with their publication years.
  
  **Dec 27, 2023**: Initial list with first 15 papers.

</details>

<br>


## Survey:
### 1. Generative AI meets 3D: A Survey on Text-to-3D in AIGC Era [Arxiv, 2023]

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



<br>

## Image to 3D Generation:
### 1. Zero-1-to-3: Zero-shot One Image to 3D Object

**Authors**: Ruoshi Liu1, Rundi Wu1, Basile Van Hoorick1, Pavel Tokmakov2, Sergey Zakharov2, Carl Vondrick1

<details span>
<summary><b>Abstract</b></summary>
We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.
</details>

  [📄 Paper](https://arxiv.org/abs/2303.11328) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/cvlab-columbia/zero123) | [🤗 Hugging Face](https://huggingface.co/spaces/cvlab/zero123-live)

### 2. Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model

**Authors**: Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, Hao Su

<details span>
<summary><b>Abstract</b></summary>
We report Zero123++, an image-conditioned diffusion model for generating 3D-consistent multi-view images from a single input view. To take full advantage of pretrained 2D generative priors, we develop various conditioning and training schemes to minimize the effort of finetuning from off-the-shelf image diffusion models such as Stable Diffusion. Zero123++ excels in producing high-quality, consistent multi-view images from a single image, overcoming common issues like texture degradation and geometric misalignment. Furthermore, we showcase the feasibility of training a ControlNet on Zero123++ for enhanced control over the generation process. The code is available at this https URL.
</details>

  [📄 Paper](https://arxiv.org/abs/2310.15110) | [🌐 Project Page](https://zero123.cs.columbia.edu/) | [💻 Code](https://github.com/SUDO-AI-3D/zero123plus) | [🤗 Hugging Face](https://huggingface.co/spaces/sudo-ai/zero123plus-demo-space)



<br>

## Editing:
### 1. DreamEditor: Text-Driven 3D Scene Editing with Neural Fields

**Authors**: Zhuang, Jingyu and Wang, Chen and Liu, Lingjie and Lin, Liang and Li, Guanbin

<details span>
<summary><b>Abstract</b></summary>
Neural fields have achieved impressive advancements in view synthesis and scene reconstruction. However, editing these neural fields remains challenging due to the implicit encoding of geometry and texture information. In this paper, we propose DreamEditor, a novel framework that enables users to perform controlled editing of neural fields using text prompts. By representing scenes as mesh-based neural fields, DreamEditor allows localized editing within specific regions. DreamEditor utilizes the text encoder of a pretrained text-to-Image diffusion model to automatically identify the regions to be edited based on the semantics of the text prompts. Subsequently, DreamEditor optimizes the editing region and aligns its geometry and texture with the text prompts through score distillation sampling [Poole et al. 2022]. Extensive experiments have demonstrated that DreamEditor can accurately edit neural fields of real-world scenes according to the given text prompts while ensuring consistency in irrelevant areas. DreamEditor generates highly realistic textures and geometry, significantly surpassing previous works in both quantitative and qualitative evaluations.
</details>

  [📄 Paper](https://arxiv.org/abs/2306.13455) | [🌐 Project Page](https://www.sysu-hcp.net/projects/cv/111.html) | [💻 Code](https://github.com/zjy526223908/DreamEditor) 


### 2. IDE-3D: Interactive Disentangled Editing For High-Resolution 3D-aware Portrait Synthesis

**Authors**: Jingxiang Sun, Xuan Wang, Yichun Shi, Lizhen Wang, Jue Wang, Yebin Liu

<details span>
<summary><b>Abstract</b></summary>
Existing 3D-aware facial generation methods face a dilemma in quality versus editability: they either generate editable results in low resolution, or high quality ones with no editing flexibility. In this work, we propose a new approach that brings the best of both worlds together. Our system consists of three major components: (1) a 3D-semantics-aware generative model that produces view-consistent, disentangled face images and semantic masks; (2) a hybrid GAN inversion approach that initialize the latent codes from the semantic and texture encoder, and further optimized them for faithful reconstruction; and (3) a canonical editor that enables efficient manipulation of semantic masks in canonical view and producs high quality editing results. Our approach is competent for many applications, e.g. free-view face drawing, editing and style control. Both quantitative and qualitative results show that our method reaches the state-of-the-art in terms of photorealism, faithfulness and efficiency.
</details>

  [📄 Paper](https://arxiv.org/abs/2205.15517) | [🌐 Project Page](https://mrtornado24.github.io/IDE-3D/) | [💻 Code](https://github.com/MrTornado24/IDE-3D) 



### 3. DM-NeRF: 3D Scene Geometry Decomposition and Manipulation from 2D Images

**Authors**: Bing Wang, Lu Chen, Bo Yang

<details span>
<summary><b>Abstract</b></summary>
 In this paper, we study the problem of 3D scene geometry decomposition and manipulation from 2D views. By leveraging the recent implicit neural representation techniques, particularly the appealing neural radiance fields, we introduce an object field component to learn unique codes for all individual objects in 3D space only from 2D supervision. The key to this component is a series of carefully designed loss functions to enable every 3D point, especially in non-occupied space, to be effectively optimized even without 3D labels. In addition, we introduce an inverse query algorithm to freely manipulate any specified 3D object shape in the learned scene representation. Notably, our manipulation algorithm can explicitly tackle key issues such as object collisions and visual occlusions. Our method, called DM-NeRF, is among the first to simultaneously reconstruct, decompose, manipulate and render complex 3D scenes in a single pipeline. Extensive experiments on three datasets clearly show that our method can accurately decompose all 3D objects from 2D views, allowing any interested object to be freely manipulated in 3D space such as translation, rotation, size adjustment, and deformation.
</details>

  [📄 Paper](https://arxiv.org/abs/2208.07227) | [💻 Code](https://github.com/vLAR-group/DM-NeRF) 

<br>

## Human-Avatar Generation:
### 1. AvatarBooth: High-Quality and Customizable 3D Human Avatar Generation

**Authors**: Yifei Zeng1, Yuanxun Lu1, Xinya Ji1, Yao Yao1, Hao Zhu1, Xun Cao1,

<details span>
<summary><b>Abstract</b></summary>
 We introduce AvatarBooth, a novel method for generating high-quality 3D avatars using text prompts or specific images. Unlike previous approaches that can only synthesize avatars based on simple text descriptions, our method enables the creation of personalized avatars from casually captured face or body images, while still supporting text-based model generation and editing. Our key contribution is the precise avatar generation control by using dual fine-tuned diffusion models separately for the human face and body. This enables us to capture intricate details of facial appearance, clothing, and accessories, resulting in highly realistic avatar generations. Furthermore, we introduce pose-consistent constraint to the optimization process to enhance the multi-view consistency of synthesized head images from the diffusion model and thus eliminate interference from uncontrolled human poses. In addition, we present a multi-resolution rendering strategy that facilitates coarse-to-fine supervision of 3D avatar generation, thereby enhancing the performance of the proposed system. The resulting avatar model can be further edited using additional text descriptions and driven by motion sequences. Experiments show that AvatarBooth outperforms previous text-to-3D methods in terms of rendering and geometric quality from either text prompts or specific images. Please check our project website at this https URL.
</details>

  [📄 Paper](https://arxiv.org/pdf/2306.09864) | [🌐 Project Page](https://zeng-yifei.github.io/avatarbooth_page/) | [💻 Code](https://github.com/zeng-yifei/AvatarBooth) 



### 2. SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance

**Authors**: Yuanyou Xu, Zongxin Yang, Yi Yang

<details span>
<summary><b>Abstract</b></summary>
 Powered by large-scale text-to-image generation models, text-to-3D avatar generation has made promising progress. However, most methods fail to produce photorealistic results, limited by imprecise geometry and low-quality appearance. Towards more practical avatar generation, we present SEEAvatar, a method for generating photorealistic 3D avatars from text with SElf-Evolving constraints for decoupled geometry and appearance. For geometry, we propose to constrain the optimized avatar in a decent global shape with a template avatar. The template avatar is initialized with human prior and can be updated by the optimized avatar periodically as an evolving template, which enables more flexible shape generation. Besides, the geometry is also constrained by the static human prior in local parts like face and hands to maintain the delicate structures. For appearance generation, we use diffusion model enhanced by prompt engineering to guide a physically based rendering pipeline to generate realistic textures. The lightness constraint is applied on the albedo texture to suppress incorrect lighting effect. Experiments show that our method outperforms previous methods on both global and local geometry and appearance quality by a large margin. Since our method can produce high-quality meshes and textures, such assets can be directly applied in classic graphics pipeline for realistic rendering under any lighting condition. Project page at: this https URL.
</details>

  [📄 Paper](https://arxiv.org/pdf/2312.08889.pdf) | [🌐 Project Page](https://seeavatar3d.github.io/) 



### 3. Text and Image Guided 3D Avatar Generation and Manipulation

**Authors**: Zehranaz Canfes, M. Furkan Atasoy, Alara Dirik, Pinar Yanardag

<details span>
<summary><b>Abstract</b></summary>
 The manipulation of latent space has recently become an interesting topic in the field of generative models. Recent research shows that latent directions can be used to manipulate images towards certain attributes. However, controlling the generation process of 3D generative models remains a challenge. In this work, we propose a novel 3D manipulation method that can manipulate both the shape and texture of the model using text or image-based prompts such as 'a young face' or 'a surprised face'. We leverage the power of Contrastive Language-Image Pre-training (CLIP) model and a pre-trained 3D GAN model designed to generate face avatars, and create a fully differentiable rendering pipeline to manipulate meshes. More specifically, our method takes an input latent code and modifies it such that the target attribute specified by a text or image prompt is present or enhanced, while leaving other attributes largely unaffected. Our method requires only 5 minutes per manipulation, and we demonstrate the effectiveness of our approach with extensive results and comparisons.
</details>

  [📄 Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Canfes_Text_and_Image_Guided_3D_Avatar_Generation_and_Manipulation_WACV_2023_paper.pdf) | [🌐 Project Page](https://catlab-team.github.io/latent3D/) | [💻 Code](https://github.com/catlab-team/latent3D_code) 


<br>

## Autonomous Driving:

### 1. 3D Object Detection for Autonomous Driving: A Comprehensive Survey

**Authors**: Jiageng Mao, Shaoshuai Shi, Xiaogang Wang, Hongsheng Li

<details span>
<summary><b>Abstract</b></summary>
 Autonomous driving, in recent years, has been receiving increasing attention for its potential to relieve drivers' burdens and improve the safety of driving. In modern autonomous driving pipelines, the perception system is an indispensable component, aiming to accurately estimate the status of surrounding environments and provide reliable observations for prediction and planning. 3D object detection, which intelligently predicts the locations, sizes, and categories of the critical 3D objects near an autonomous vehicle, is an important part of a perception system. This paper reviews the advances in 3D object detection for autonomous driving. First, we introduce the background of 3D object detection and discuss the challenges in this task. Second, we conduct a comprehensive survey of the progress in 3D object detection from the aspects of models and sensory inputs, including LiDAR-based, camera-based, and multi-modal detection approaches. We also provide an in-depth analysis of the potentials and challenges in each category of methods. Additionally, we systematically investigate the applications of 3D object detection in driving systems. Finally, we conduct a performance analysis of the 3D object detection approaches, and we further summarize the research trends over the years and prospect the future directions of this area.
</details>

  [📄 Paper]() | [💻 Code](https://github.com/PointsCoder/Awesome-3D-Object-Detection-for-Autonomous-Driving) 


### 2. A Survey on Safety-Critical Driving Scenario Generation – A Methodological Perspective

**Authors**: Wenhao Ding, Chejian Xu, Mansur Arief, Haohong Lin, Bo Li, Ding Zhao

<details span>
<summary><b>Abstract</b></summary>
 Autonomous driving systems have witnessed a significant development during the past years thanks to the advance in machine learning-enabled sensing and decision-making algorithms. One critical challenge for their massive deployment in the real world is their safety evaluation. Most existing driving systems are still trained and evaluated on naturalistic scenarios collected from daily life or heuristically-generated adversarial ones. However, the large population of cars, in general, leads to an extremely low collision rate, indicating that the safety-critical scenarios are rare in the collected real-world data. Thus, methods to artificially generate scenarios become crucial to measure the risk and reduce the cost. In this survey, we focus on the algorithms of safety-critical scenario generation in autonomous driving. We first provide a comprehensive taxonomy of existing algorithms by dividing them into three categories: data-driven generation, adversarial generation, and knowledge-based generation. Then, we discuss useful tools for scenario generation, including simulation platforms and packages. Finally, we extend our discussion to five main challenges of current works -- fidelity, efficiency, diversity, transferability, controllability -- and research opportunities lighted up by these challenges.
</details>

  [📄 Paper](https://arxiv.org/abs/2202.02215) 


<br>

## BioMedical:

### 1. MinD-3D: Reconstruct High-quality 3D objects in Human Brain
**Authors**: Jianxiong Gao, Yuqian Fu, Yun Wang, Xuelin Qian, Jianfeng Feng, Yanwei Fu

<details span>
<summary><b>Abstract</b></summary>
In this paper, we introduce Recon3DMind, a groundbreaking task focused on reconstructing 3D visuals from Functional Magnetic Resonance Imaging (fMRI) signals. This represents a major step forward in cognitive neuroscience and computer vision. To support this task, we present the fMRI-Shape dataset, utilizing 360-degree view videos of 3D objects for comprehensive fMRI signal capture. Containing 55 categories of common objects from daily life, this dataset will bolster future research endeavors. We also propose MinD-3D, a novel and effective three-stage framework that decodes and reconstructs the brain's 3D visual information from fMRI signals. This method starts by extracting and aggregating features from fMRI frames using a neuro-fusion encoder, then employs a feature bridge diffusion model to generate corresponding visual features, and ultimately recovers the 3D object through a generative transformer decoder. Our experiments demonstrate that this method effectively extracts features that are valid and highly correlated with visual regions of interest (ROIs) in fMRI signals. Notably, it not only reconstructs 3D objects with high semantic relevance and spatial similarity but also significantly deepens our understanding of the human brain's 3D visual processing capabilities. Project page at:  https://jianxgao.github.io/MinD-3D.
</details>

  [📄 Paper](https://arxiv.org/abs/2312.07485) | [🌐 Project Page](https://jianxgao.github.io/MinD-3D/) | [💻 Code](https://github.com/JianxGao/MinD-3D) 



<be>

## 4D AIGC


### 1. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency


**Authors**: Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, Yunchao Wei

<details span>
<summary><b>Abstract</b></summary>
 Aided by text-to-image and text-to-video diffusion models, existing 4D content creation pipelines utilize score distillation sampling to optimize the entire dynamic 3D scene. However, as these pipelines generate 4D content from text or image inputs, they incur significant time and effort in prompt engineering through trial and error. This work introduces 4DGen, a novel, holistic framework for grounded 4D content creation that decomposes the 4D generation task into multiple stages. We identify static 3D assets and monocular video sequences as key components in constructing the 4D content. Our pipeline facilitates conditional 4D generation, enabling users to specify geometry (3D assets) and motion (monocular videos), thus offering superior control over content creation. Furthermore, we construct our 4D representation using dynamic 3D Gaussians, which permits efficient, high-resolution supervision through rendering during training, thereby facilitating high-quality 4D generation. Additionally, we employ spatial-temporal pseudo labels on anchor frames, along with seamless consistency priors implemented through 3D-aware score distillation sampling and smoothness regularizations. Compared to existing baselines, our approach yields competitive results in faithfully reconstructing input signals and realistically inferring renderings from novel viewpoints and timesteps. Most importantly, our method supports grounded generation, offering users enhanced control, a feature difficult to achieve with previous methods. Project page: this https URL
</details>

  [📄 Paper](https://arxiv.org/abs/2312.17225) | [🌐 Project Page](https://vita-group.github.io/4DGen/) | [💻 Code](https://github.com/VITA-Group/4DGen) 



<br>


<!--
## Misc:

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
