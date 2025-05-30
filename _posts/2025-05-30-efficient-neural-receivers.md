---
layout: distill
title: "Accelerating Efficient Neural Receivers for Real-Time 5G Communication: Methods and Implementation"
description: EPIC Labs
date: 2025-05-30
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Seungjun Kim
    url: "https://sites.google.com/view/epiclab/member/sjkim"
    affiliations:
      name: Pohang University of Science and Technology
  - name: Soonhyun Kwon
    url: "https://sites.google.com/view/epiclab/member/shkwon"
    affiliations:
      name: Pohang University of Science and Technology
  - name: Chanhee Lee
    url: "https://sites.google.com/view/epiclab/member/chlee"
    affiliations:
      name: Pohang University of Science and Technology

# must be the exact same name as your blogpost
bibliography: 2025-05-30-efficient-neural-receivers.bib

toc:
  - name: "Introduction"
  - name: "Preliminaries and Neural Receiver"
  - name: "Quantized Neural Receiver: Everything About Quantization for EffiNRX"
    subsections:
    - name: Number Systems — Two Ways a Computer Represents Numbers
    - name: What Is Quantization and Why Bother?
      subsections:
      - name: Symmetric vs Asymmetric Scaling
      - name: Static vs Dynamic Scaling
      - name: PTQ (Post-Training Quantization) vs QAT (Quantization-Aware Training)
    - name: Quantize NRX for Efficient Implementation
      subsections:
      - name: Insert Explicit Q / DQ Nodes
      - name: Model compile with TensorRT
      - name: TBLER simulation
      - name: Throughput and Latency anaysis
      - name: Why is FP8 faster than INT8 even though both use 8 bits?
  - name: "Pruned Neural Receiver: Beyond Compression Toward Hardware Acceleration"
    subsections:
    - name: "Unstructured vs Structured: Which Pruning Approach Is More Effective?"
      subsections:
      - name: Unstructured Pruning
      - name: Structured Pruning
    - name: Pruning Experiment Setup for EffiNRX
    - name: "Impact of Pruning: Changes in TBLER and Acceleration"
      subsections:
      - name: NRX RT
      - name: NRX Large
    - name: "Conclusion: Is Pruning an Attractive Solution for Neural Network-Based Communication Systems?"
  - name: "Distilled Neural Receiver: Transferring Knowledge for Lightweight Inference"
    subsections:
    - name: Distillation Strategy
  - name: "Optimized Neural Receiver: Real-Time Efficiency at the Edge"
    subsections:
    - name: Additional Latency Analysis
  - name: "Conclusions"


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  d-article {
    overflow-x: visible;
  }

  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

  [data-theme="dark"] details[open] {
  --bg: #112f4a;
  color: white;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  font-size: 80%;
  }
  .box-note {
    font-size: 18px;
    padding: 15px 15px 0px 15px;
    margin: 20px 20px 20px 5px;
    border: 1px solid #eee;
    border-left-width: 5px;
    border-radius: 5px 5px 5px 5px;
  }
  d-article .box-note {
    background-color: #F8FAFC;
    border-left-color: #c80150;
  }
  html[data-theme='dark'] d-article .box-note {
    background-color: #F8FAFC;
    border-left-color: #c80150;
  }

---


# Introduction
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/system_model.png" class="img-fluid rounded z-depth-1" %}
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/NRX.png" class="img-fluid rounded z-depth-1" %}


# Preliminaries and Neural Receiver

# Quantized Neural Receiver: Everything About Quantization for EffiNRX
## Number Systems — Two Ways a Computer Represents Numbers
Before diving into deep learning, we must first understand how hardware encodes and manipulates “numbers.” Modern processors usually offer two representations: integers and floating-point.
- **Integers** store values as two’s-complement binary within a fixed bit-width. For example, 8-bit `int8` covers −128 to 127 exactly. Because each representable value is spaced uniformly, integers are ideal for evenly distributed data. Their adders and multipliers are also far simpler than floating-point units, giving big wins in power and latency. When fractional precision is required, one can move the binary “point” inside the word (fixed-point), but every new point location forces a redesign of the entire compute path and costs extra shifters.
- **Floating-point** mimics scientific notation, splitting a number into a significand and an exponent. *IEEE-754* `fp32` uses 1 sign bit, 8 exponent bits, and 23 fraction bits, covering roughly  $(10^{-38}\)~\(10^{38})$Unlike integers, the spacing between values shrinks near zero and widens further away. Because neural-network weights are roughly Gaussian, they cluster around zero; dense spacing there is a huge advantage. Floating-point is therefore indispensable during training, but full 32-bit units are bulky and hot—awkward on mobile devices. Recent formats such as `fp16` or `fp8` trim the mantissa aggressively, yet they still cost more gates than integers.

## What Is Quantization and Why Bother?
Using more bits yields arithmetic that matches real-number math, but memory and logic scale linearly (or worse) with bit-width. Doubling precision rarely doubles accuracy. Quantization purposefully reduces bit-width while capping the accuracy loss. Formally, a real value $x$ is approximated by an integer $q$ via a scale $S$ and zero-point $Z$:

$$
q \;=\; \mathrm{round}\!\left(\frac{x}{S}\right) + Z,\quad
x \approx S\,(q - Z)
$$

This brings three key benefits:

- **Model size**: Converting fp32 weights to `int8` (or `fp8`) cuts storage by 4×; int4 squeezes it by 8×.
- **Compute & power**: Narrower ALUs mean fewer gates and faster clocks, slashing energy per MAC.
- **Memory bandwidth**: Shifting from 32-bit to 8-bit cuts DRAM↔NPU traffic by 75 %.


### Symmetric vs Asymmetric Scaling
Symmetric quantization places 0 at the center of a ±max range—perfect for weight tensors whose mean is near zero. Asymmetric quantization shifts the range upward, wasting no codes on impossible negatives (e.g. ReLU activations). It needs per-multiply offset correction, so the hardware is slightly busier.

### Static vs Dynamic Scaling
Static quantization (calibration): Run a held-out sample set once, fix the min/max, and reuse the scale at inference. Simple and fast, but brittle if runtime data drift. Dynamic quantization: Re-estimate min/max per batch (or via EMA). Extra overhead, yet very effective in NLP where sequence lengths vary wildly.

### PTQ (Post-Training Quantization) vs QAT (Quantization-Aware Training)
PTQ inserts Quantize/Dequantize (Q/DQ) ops after a full-precision model has converged. A few hundred calibration batches usually suffice. `int8` PTQ keeps accuracy high on both CNNs and Transformers, but at 4 bits or less the error explodes. QAT fuses “fake-quant” nodes directly into the forward pass while letting gradients flow unaltered (Straight-Through Estimator). Accuracy even at 4 bits is excellent, yet training time grows and framework support is trickier, especially for large LLMs.


## Quantize NRX for Efficient Implementation
NVIDIA GPUs provide TensorRT, which maps operations like matrix multiplication onto optimized engines, assigning the right low-precision units and maximizing parallelism—indispensable for efficient DNN deployment on GPU. The existing NRX framework is TensorFlow-based, and its wireless-channel library Sionna is likewise. While PyTorch models can be quantized in place, TensorFlow ones must first be exported to ONNX and then quantized. Thus the workflow is:

1. Export the TensorFlow model to **ONNX**.
2. Quantize the ONNX graph.
3. Build a TensorRT engine.

In our experiment we wrap NRX’s positional-encoding and data-extraction blocks so that they take the initial channel estimates `h_hat_real/imag` and received symbols `rx_slot_real/imag`, and output LLRs. The wrapper is exported to ONNX, then quantized with TensorRT’s modelopt.onnx.quantization. The workflow has two key stages.


### Insert Explicit Q / DQ Nodes
Implicit quantization via trtexec often misses layers. Instead we invoke modelopt.onnx.quantization and force Q/DQ(Quantization/Dquantization) nodes:
```shell=
python3 -m modelopt.onnx.quantization --onnx_path=../onnx_models/nrx_large.onnx \
        --output_path=../onnx_models/nrx_large_fp8.onnx \
        --quantize_mode=fp8 \
        --calibration_data calib.npz \
        --use_zero_point True \
        --calibration_shapes="rx_slot_real:1x1584x14x4,rx_slot_imag:1x1584x14x4,h_hat_real:1x1584x2x4,h_hat_imag:1x1584x2x4"
```
The resulting `.onnx` file now contains explicit Quantize/Dequantize nodes in front of every convolution and matrix-multiplication layer, ensuring TensorRT cannot silently skip them.
    
<!-- ![qdq](https://hackmd.io/_uploads/ry3Ds6Lfeg.png) -->
<!-- {% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/NRX.png" class="img-fluid rounded z-depth-1" %} -->

In the resulting .onnx, each convolution or matmul is now preceded by Q/DQ nodes, faithfully modeling quantization effects.

### Model compile with TensorRT
With Q/DQ in place, we compile the model:
```shell=
trtexec --fp8 \
        --stronglyTyped \
        --onnx=../onnx_models/nrx_large_fp8.onnx \
        --saveEngine=../onnx_models/nrx_large_fp8.plan \
        --minShapes=rx_slot_real:1x1584x14x4,rx_slot_imag:1x1584x14x4,h_hat_real:1x1584x2x4,h_hat_imag:1x1584x2x4 \
        --optShapes=rx_slot_real:1x1584x14x4,rx_slot_imag:1x1584x14x4,h_hat_real:1x1584x2x4,h_hat_imag:1x1584x2x4 \
        --maxShapes=rx_slot_real:1x1584x14x4,rx_slot_imag:1x1584x14x4,h_hat_real:1x1584x2x4,h_hat_imag:1x1584x2x4
```
Although you can specify the quantization precision with the `--fp8` flag, TensorRT cannot correctly recognize the layers unless you also pass the `--stronglyTyped` flag to indicate that explicit quantization layers are present. This is a known issue in TensorRT 10.2. When the information is supplied correctly, TensorRT will compile the model and generate dummy inputs for testing. However, if the dummy-input dimensions are not provided accurately, layers such as Reshape inside the ONNX model will throw errors, so you must explicitly declare the input dimensions.

### TBLER simulation
Following the baseline scenario—two users, four base-station antennas, and uplink transmission from the users to the BS—we performed the quantization described above. For `fp8` we simply used the maximum-value scaling method, whereas for `int8` we adopted AWQ. The target models are NRX_RT (CGNN iterations = 2) and NRX_Large (CGNN iterations = 8).
NRX_Large achieves better TBLER thanks to its higher iteration count, but its 0.44 M parameters make it relatively large and slower at inference. NRX_RT (Real-Time) delivers lower performance than NRX_Large because it uses only two iterations, yet it is efficient with just 0.14 M parameters and enjoys faster inference. Moreover, it satisfies the URLLC latency requirement of < 1 ms.
Performance is evaluated by TBLER (Transmit Block Error Rate), i.e., the fraction of transmitted blocks that contain errors; TBLER generally decreases as noise power is reduced

<!-- ![Quant_TBLER](https://hackmd.io/_uploads/SJXxZ4UMex.png) -->
<!-- {% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/NRX.png" class="img-fluid rounded z-depth-1" %} -->

In the figure, the left graph shows RT and the right graph shows Large. To benchmark quantization, we plotted three additional curves:

- LSlin + LMMSE – LS (Least Square) channel estimation with LMMSE (Linear Minimum Mean Square Error) detection; the most basic receiver.
- LMMSE + K-best – LMMSE channel estimation with K-best detection; powerful performance but extremely high complexity due to complex operations such as matrix inversion.
- Perf. CSI + K-best – K-best detection with perfect CSI; effectively ground truth.

For RT, baseline `fp16` is a little worse than LMMSE + K-best. Applying `fp8` quantization causes a noticeable drop yet still outperforms the basic receiver, whereas `int8` quantization degrades performance to an unusable level. This implies that RT’s parameter distribution is heavily concentrated near the zero-point with high variance and outliers, making `fp8` quantization more effective than `int8`.
In the Large model, the `fp16` curve is already slightly better than LMMSE + K-best. `fp8` introduces only a tiny loss and still beats LMMSE + K-best; `int8` loses a bit more and ends up roughly equal to LMMSE + K-best. Because Large has far more parameters than RT, its accuracy is less sensitive to quantization. In both cases, FP8 surpasses `int8` thanks to the weight-distribution characteristics.

### Throughput and Latency anaysis
As noted earlier, achieving a low error rate is critical; however, latency is equally vital—if processing exceeds the time budget, the communication protocol collapses. We therefore analyzed latency on an RTX 6000 Ada.

<center>

|         Model         | RT_FP32 |   RT_FP16    |    RT_FP8    |   RT_INT8    |
|:---------------------:|:-------:|:------------:|:------------:|:------------:|
| **Throughput [fps]**  | 1029.74 | **1930.18**  | **2757.71**  | **2249.35**  |
| **Throughput [Mbps]** | 182.68  |  **342.43**  |  **489.24**  |  **399.05**  |
|   **Latency [ms**]    | 1.17334 | **0.670654** | **0.495239** | **0.579834** |

</center>

For RT, `fp32` exceeds the URLLC 1 ms budget (1.173 ms), but `fp16`, `fp8`, and `int8` all meet it. The baseline paper, run on an A100, reported 1 ms for RT_FP16; our numbers differ because we use the Ada Lovelace rather than the Ampere architecture—details to follow. Relative to `fp16`, `fp8` and `int8` achieve speed-ups of *×1.27* and *×1.13*, respectively.

<center>

|         Model         | Large_FP32 | Large_FP16 | Large_FP8 | Large_INT8 |
|:---------------------:|:----------:|:----------:|:---------:|:----------:|
| **Throughput [fps]**  |  337.842   |  600.113   |  938.074  |   669.98   |
| **Throughput [Mbps]** |   59.94    |   106.46   |  166.42   |   118.86   |
|   **Latency [ms]**    |  3.23242   |  1.86426   |  1.24255  |  1.68542   |

</center>

Large shows similar trends:`fp8` and `int8` deliver ×1.33 and ×1.11 speed-ups over `fp16`. TensorRT 10.2’s `fp8` quantization focuses mainly on MatMul, so it is not ideal for NRX, which relies heavily on convolutions. All fully connected layers were quantized, but in separable convolutions only the pointwise-convolution (channel-wise fully connected) part was quantized; depthwise convolutions were not.

### Why is FP8 faster than INT8 even though both use 8 bits?
> - `fp8` needs no type conversion during accumulation.
> - `int8` incurs dequantization/requantization overhead to `fp16`.
> - On Tensor Cores, `fp8` operations can be fused into a single kernel.

<aside class="l-body box-note" markdown="1">
For these reasons, `fp8` enables faster NRX inference than `int8`—an effect that generalizes to other models. For example, with Diffusion XL 1.0, `int8` and `fp8` achieve ×1.72 and ×1.95 speed-ups over `fp16`, respectively.

In summary, `fp8` quantization minimizes accuracy loss while offering the best speed-up, but even so, none of the precisions fully satisfy every URLLC requirement, so additional compression techniques or model modifications are still needed.
</aside>


# Pruned Neural Receiver: Beyond Compression Toward Hardware Acceleration
Despite their impressive performance, modern deep learning models often require substantial computational resources and memory due to their massive number of parameters. This becomes a significant obstacle in resource constrained environments such as mobile devices and edge platforms. One effective approach to address this issue is pruning, a technique that selectively removes less important parameters within a neural network to reduce model size and improve computational efficiency.

The core idea of pruning goes beyond merely removing unnecessary parameters it lies in discovering an optimal subnetwork within a larger model. This concept is closely tied to the well known Lottery Ticket Hypothesis<d-cite key="pruning"></d-cite>. According to this hypothesis, large neural network contains smaller subnetworks referred to as ***winning lottery tickets*** that can achieve comparable or even superior performance to the original network. Thus, the pruning process can be viewed not just as a form of compression, but as a means of uncovering a hidden, efficient architecture embedded within the overparameterized model.
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/pruning.png" class="img-fluid rounded z-depth-1" %}

Therefore, pruning is a process that increases the sparsity of models by removing redundant parameters from large networks without significantly compromising accuracy. By identifying and preserving subnetworks that maintain or even surpass the performance of original model, pruning allows neural networks to become lighter and faster while retaining their effectiveness. Through this technique, sparsity is improved with minimal degradation in performance, making the models more suitable for deployment on resource-constrained devices.
The main advantages of applying pruning are as follows:
1. <strong>Model Size Reduction</strong>: By removing unimportant parameters, pruning can significantly reduce the storage requirements of a model. When combined with formats like CSR (Compressed Sparse Row) or CSC (Compressed Sparse Column)—which are efficient for storing sparse matrices—models can avoid storing unnecessary weights. This is especially valuable in memory-constrained environments such as embedded systems. However, since sparse formats require additional metadata (e.g., indices), their benefits are typically realized only when the sparsity exceeds 50%.
2. <strong>Hardware Acceleration</strong>: Pruned weights are effectively zero, meaning the corresponding multiplications and additions can be skipped during computation. Furthermore, fewer weights need to be loaded from memory, which helps alleviate the memory bandwidth bottleneck that often limits performance in large-scale neural networks. These advantages make it possible to leverage hardware optimized for sparse computation—such as sparse engines or specialized accelerators—to achieve significantly faster inference.
3. <strong>Potential for Improved Generalization:</strong>: By eliminating unnecessary parameters, pruning encourages the model to focus on learning only the most essential features from the data. This reduction in complexity can help mitigate overfitting, especially in overparameterized networks, and in some cases, even lead to improved validation performance. Pruning thus not only compresses the model but can also act as an implicit regularizer, enhancing its ability to generalize to unseen data.

A common strategy in pruning methods is to retain only the top $v\%$ of weights where $v = 1 - \text{sparsity}$ based on a scoring criterion assigned to each weight. We define the $Top_v$ function as a selector that preserves the weights corresponding to the top $v%$ values in the score matrix $S$.

$$
\text{Top}_v(\mathbf{S})_{i,j} = 
\begin{cases}
1, & \text{if } S_{i,j} \text{ is in top } v\% \\
0, & \text{o.w.}
\end{cases}
$$


## Unstructured vs Structured: Which Pruning Approach Is More Effective?
In this way, pruning has established itself as a powerful technique that goes beyond simple model compression enabling the construction of more efficient and lightweight models while also facilitating hardware acceleration and performance improvement. In the following sections, we provide a detailed introduction to the two main pruning approaches: Unstructured Pruning and Structured Pruning.


### Unstructured Pruning
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/Unstructed.png" class="img-fluid rounded z-depth-1" %}
Unstructured pruning removes individual weights based on their importance typically determined by their magnitude. For instance, weights with absolute values below a certain threshold can be pruned away. This approach allows for the highest levels of sparsity with minimal performance degradation, making it one of the most effective methods for reducing model size. However, because the resulting sparsity is irregular and scattered, it is difficult to exploit for acceleration on standard dense matrix hardware. Leveraging unstructured pruning for runtime speedup usually requires specialized sparse matrix libraries or dedicated sparse engines. Common methods for unstructured pruning include magnitude pruning and movement pruning<d-cite key="movement_pruning"></d-cite>. 

The image shown illustrates a hypothetical 0.5 sparsity unstructured pruning mask applied to a (16 X 16) weight matrix. As can be seen, the pruned weights are distributed randomly without any fixed pattern.

- Magnitude Pruning
Magnitude pruning is one of the most intuitive and widely used pruning techniques. In this method, weights are pruned after model training, based on the magnitude of each weight. Specifically, weights with small absolute values are considered less important and are removed under the assumption that *weights with smaller magnitudes contribute less to the model's output, so removing them will not significantly degrade performance.*
This approach is officially supported by the `tensorflow_model_optimization` package, making it accessible and easy to implement.
While simple and effective in many cases, magnitude pruning has its limitations. A small weight at one point in training might later become critical to model performance. Therefore, relying solely on magnitude can sometimes lead to sub-optimal pruning results, especially if done without careful scheduling or fine-tuning.
The score $S$ for magnotude pruning is defined as follows:

$$
S_{i,j} = abs(w_{i,j})
$$

- Movement Pruning<d-cite key="movement_pruning"></d-cite>
Movement pruning was proposed to overcome the limitations of magnitude pruning. Rather than relying solely on the current magnitude of weights, this method evaluates weight importance based on how much a weight changes during training.
Unlike magnitude pruning, which only considers the absolute value of a weight, movement pruning tracks the trajectory of each weight during training—especially during the fine-tuning phase. If a weight consistently moves away from zero, it is considered important for learning. Conversely, even if a weight has a large magnitude, it may be deemed unimportant if it trends toward zero or shows little change throughout training. The measure of this “movement” can be based purely on the weight update history or incorporate gradient information from the loss function.
By capturing a weight’s dynamic contribution during training rather than just its static value, movement pruning provides a more nuanced assessment of importance. This often leads to more optimal pruning results and better model performance compared to magnitude-based methods. However, since it requires tracking weight updates over time, its implementation is more complex and incurs a higher computational cost than magnitude pruning.
The score $S$ at $T$ step for movement pruning is defined as follows:

$$
S_{i,j}^{(T)} = -\alpha_S \sum_{t < T} \left( \frac{\partial \mathcal{L}}{\partial W_{i,j}} \right)^{(t)} W_{i,j}^{(t)}
$$

Both of these methods magnitude and movement pruning are forms of unstructured pruning, which results in sparse masks with irregular patterns. While effective at reducing model size, this irregularity makes it difficult to accelerate inference on most hardware. To address this, a different class of methods called structured pruning introduces regular sparsity patterns that enable hardware-friendly acceleration.

### Structured Pruning
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/Structed.png" class="img-fluid rounded z-depth-1" %}

- Common Method
Structured pruning removes groups of weights such as entire channels, filters, or neurons instead of individual connections. While this approach may achieve lower sparsity compared to unstructured pruning, however this makes it highly hardware-friendly and easier to integrate with existing deep learning frameworks.
In Transformer-based large language models (LLMs) like GPT, pruning can be applied to vector structures, leveraging the inherent structure of weights. The weights of pre-trained Transformers often exhibit vectorized patterns, making vector-structured pruning naturally aligned with the underlying architecture. Studies have shown that combining vector-level structured pruning with hardware optimized for vector utilization can lead to significantly improved throughput<d-cite key="tfmvp"></d-cite>.

- 2:4 Method
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/24_Structed.png" class="img-fluid rounded z-depth-1" %}

Recent commercial GPU architectures have increasingly incorporated dedicated hardware support for acceleration through pruning. A prominent example is NVIDIA’s Ampere architecture, which introduces a specialized hardware component known as the ***Sparse Tensor Core***.<d-cite key="nvidia_sparse_tensor_core"></d-cite>
According to the NVIDIA Developer Blog post titled "Accelerating Inference with Sparsity Using Ampere and TensorRT"<d-cite key="pruning_rt"></d-cite>, the Ampere architecture improves performance by exploiting a 2:4 sparsity pattern during matrix multiplication operations. This pattern requires that within every group of four consecutive elements, only two can be non-zero. In other words, for a tensor to qualify for sparse acceleration, each 4-element block must contain at most two non-zero values, enabling the hardware to skip redundant computations and achieve faster inference.
NVIDIA GPUs equipped with ***Sparse Tensor Cores*** such as A100 and RTX 6000Ada can detect this pattern and skip computations involving zero values, executing operations only on the effective non-zero elements. This allows for up to 2× improvement in throughput, making inference significantly more efficient.
The technology is integrated with NVIDIA’s deep learning inference optimization library, `TensorRT`, enabling developers to easily take advantage of sparse acceleration. Using packages like `tensorflow_model_optimization`, developers can convert pre-trained model weights to the 2:4 sparsity pattern and optimize them with TensorRT for accelerated inference on NVIDIA Ampere and newer GPUs.

{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/sparsity-improvements.png" class="img-fluid rounded z-depth-1" %}<d-footnote>Sparsity improvements in performance and power efficiency (with dense as a baseline) https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/</d-footnote>"
It has been shown that applying 2:4 pruning to ResNeXt can improve performance per watt by over 20%, while also achieving more than a 10% reduction in latency.


## Pruning Experiment Setup for EffiNRX
Pruning of the Neural Receiver was conducted in a `TensorFlow` environment. The `tensorflow_model_optimization` package provides various model optimization solutions for models built with `TensorFlow`, including support for pruning. However, since the package only supports the most basic method <strong>magnitude pruning</strong> we applied pruning to the Neural Receiver using the `prune_low_magnitude` method followed by fine-tuning.
Pruning was applied only to the weights, excluding the biases. The `prune_low_magnitude` method was used to prune all weights within the `Dense` and `SeparableConv2D` layers that make up the Neural Receiver. Since biases were not pruned, the overall model sparsity is slightly lower than the target sparsity.

Both Neural Receiver Large and RT were evaluated using <strong>0.8 sparsity unstructed magnitude pruning</strong> and <strong>2:4 structed magnitude pruning</strong>, allowing for a comprehensive assessment of model performance and pruning effectiveness under each approach.

The mask $M$ generated by applying $Top_v$ to the score matrix $S$ is used to perform an element-wise product with the weights, resulting in pruned weights that are then used in the layer's computations.

$$
\mathbf{A} = (\mathbf{W} \circ \mathbf{M}) \mathbf{X}
$$

In pruning with `tensorflow_model_optimization`, pruning_config 1) was applied to achieve 0.8 sparsity pruning, while the pruning_config 2) was used for 2:4 structured pruning.

```python=
# 1) Unstructed Pruning
pruning_config = {
    'pruning_schedule': pruning_schedule.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.8,
        begin_step=0,
        end_step=10000,
        power=1
    )
}

# 2) 2:4 Pruning
pruning_config = {
    'sparsity_m_by_n': (2, 4),
}
```


The specified `pruning_config` is applied to the pruning layer instances, successfully enabling the application of the target pruning method.

```python=
# Magnitude Pruning Layer
# layer = Dense or SeparableConv2D
layers = prune_low_magnitude(layer(d_s, (k, k), activation=None or relu, padding='same', dtype=dtype, use_bias=en_bias), **pruning_config)
```

Fine-tuning for both NRX RT and Large models was conducted under identical conditions, using only 1/1000 of the epochs used during model pre-training.

```shell=
trtexec --sparsity=enable
```

By enabling the corresponding option, `TensorRT` is able to analyze the given weights and apply appropriate sparse features to enable acceleration.

## Impact of Pruning: Changes in TBLER and Acceleration
After performing pruning in `TensorFlow`, we simulated the TBLER of the pruned NRX across an Eb/N0 range of −2 to 7 dB. In the context of Eb/N0 which represents the signal power, a lower TBLER at the same Eb/N0 indicates a more powerfull model.

We also measured GPU latency in a *float16* environment using `TensorRT`. In our experiments, even though we applied 2:4 structured pruning to enable ***Sparse Tensor Core***, we observed that some layers were still executed using dense engines.

Specifically, the `Dense` and depthwise components of `SeparableConv2D` layers in the NRX were always executed with the dense engine. For NRX RT, these layers have dimensions of (64 X 56), which are relatively small. In such cases, the overhead introduced by ***Sparse Tensor Core*** outweighed its benefits, leading to performance degradation compared to dense execution. More detailed experiments have shown that ***Sparse Tensor Core*** is highly effective in layers with large weight dimensions <d-cite key="nvidia_sparse_tensor_core"></d-cite>, where the benefits of sparsity outweigh the overhead. The same behavior was observed in NRX Large, which simply increases the number of CGNN iterations and therefore retains the same small dimensional layers.


### NRX RT
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/TBLER_pRT.png" class="img-fluid rounded z-depth-1" %}

The figure above shows the TBLER simulation results for NRX RT, including comparisons among the baseline, unstructured pruned, and 2:4 structured pruned models. Although the performance is inferior to traditional mathematical models like LMMSE + K-best, the NRX RT still **greatly surpass LS + LMMSE** in terms of error correction capability. However, regardless of the pruning method used, **pruned NRX RT exhibits a consistent performance degradation of around 2 dB**. This result stands in stark contrast to the findings observed in the NRX Large model, which demonstrates a different trend.

{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/RT_pruning_summary.png" class="img-fluid rounded z-depth-1" %}

As expected, although pruning was conducted with a target sparsity of 0.8, the actual achieved sparsity was slightly lower around 0.66 due to the exclusion of biases from pruning. Nevertheless, the pruned weights, when compressed using the CSR (Compressed Sparse Row) format, resulted in a **significant reduction in the weight file size**.

The GPU latency measured using `TensorRT` showed almost no difference across the three cases. This aligns with our previous observation: unstructured pruning cannot leverage sparse features for acceleration, while in the case of 2:4 structured pruning, the layer dimensions are too small to benefit significantly. The overhead associated with sparse execution outweighs its advantages at such scales, resulting in **negligible latency improvement**.

### NRX Large
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/TBLER_pLarge.png" class="img-fluid rounded z-depth-1" %}

In the case of NRX Large, unlike NRX RT, pruning resulted in minimal performance degradation. This is likely because NRX Large is overparameterized compared to NRX RT, containing greater redundancy, which makes it more resilient to pruning. However, the extent of performance loss varied depending on the pruning method used. In particular, the 2:4 structured pruning showed more significant degradation. This is because the structural constraint imposed for hardware acceleration prevents optimal selection of weights to prune. If a group of four weights contains three important ones, one must still be pruned to satisfy the 2:4 pattern, inevitably leading to a drop in performance.

As highlighted in the Lottery Ticket Hypothesis<d-cite key="pruning"></d-cite>, we observed that the NRX Large model, despite having fewer total parameters after unstructured pruning compared to NRX RT, achieved superior TBLER performance. This demonstrates that identifying partial ***Winning lottery tickets***is also feasible within neural network based communication models.

{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/Large_pruning_summary.png" class="img-fluid rounded z-depth-1" %}

Similarly, the weight file size was significantly reduced in the unstructured pruning case, while the 2:4 structured pruning showed limited compression benefits due to its relatively low sparsity, making it difficult to exploit the advantages of CSR formatting. GPU latency results were consistent with those of NRX RT, showing negligible differences across the configurations.


## Conclusion: Is Pruning an Attractive Solution for Neural Network-Based Communication Systems?
<aside class="l-body box-note" markdown="1">
In conclusion, the impact of pruning observed in this study was minimal, with benefits largely limited to modest reductions in storage requirements. Particularly in terms of key performance metrics for real-world communication systems such as latency and accuracy, pruning introduced overhead from fine-tuning without delivering meaningful gains. As such, applying pruning to NRX models appears to be impractical, and it is more reasonable to avoid it in deployment or optimization scenarios. Compared to quantization, pruning offered little to no advantage. Instead, exploring alternative strategies such as architectural redesign, hardware-aware optimization, or knowledge distillation may provide more substantial efficiency improvements.
</aside>



# Distilled Neural Receiver: Transferring Knowledge for Lightweight Inference
Knowledge Distillation is a model compression technique that transfers the learned behavior of a large, accurate **teacher** model to a smaller **student** model. Instead of training the student solely on ground-truth labels, it learns to mimic the soft targets—typically, the probabilistic outputs or logits—generated by the teacher. These soft targets contain richer information such as uncertainty and inter-class similarity, enabling the student to generalize better.
NRX faces a trade-off between performance and real-time inference latency. While NRX_Large model offers high performance, its inference latency exceeds 1 ms, making it impractical for URLLC applications. Therefore, lightweight models such as NRX_RT are required in latency-critical scenarios.


## Distillation Strategy
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/distill.png" class="img-fluid rounded z-depth-1" %}


{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/loss.png" class="img-fluid rounded z-depth-1" %}


{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/TBLER_distill.png" class="img-fluid rounded z-depth-1" %}



# Optimized Neural Receiver: Real-Time Efficiency at the Edge
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/TBLER_v2.png" class="img-fluid rounded z-depth-1" %} -->
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/efficiency.png" class="img-fluid rounded z-depth-1" %}

{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/Image_BER.png" class="img-fluid rounded z-depth-1" %}


## Additional Latency Analysis
<!-- We conducted additional experiments under conditions as consistent as possible with those of the previous work <d-cite key="NNNRRRXXX"></d-cite>. The focus was on evaluating whether latency improvements could be achieved through quantization and pruning techniques on both the NVIDIA A100 and RTX 6000 Ada GPUs. -->
We conducted additional experiments under conditions as consistent as possible with those of the previous work <d-cite key="NNNRRRXXX"></d-cite>. The focus was on evaluating whether latency improvements could be achieved through quantization and pruning techniques on both the NVIDIA A100 and RTX 6000 Ada GPUs.

For quantization, we examined performance using FP16, FP8, and INT8 precision. In parallel, we applied both Unstructured and 2:4 Structured pruning methods to analyze latency trends across combinations.

{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/6000ada_a100.png" class="img-fluid rounded z-depth-1" %}

Across all scenarios, pruning—regardless of type—did not lead to any meaningful improvements in inference latency. In contrast, quantization yielded clear latency reductions on the RTX 6000 Ada, with performance improving in the order of FP8, INT8, then FP16. However, since the A100 does not officially support FP8, that configuration was excluded from its experiments.

Interestingly, the RTX 6000 Ada consistently outperformed the A100 in latency by approximately 1.7× across all cases. This is primarily because NRX, unlike Transformer-based large language models, has relatively few parameters and thus does not fully leverage the A100’s HBM memory bandwidth advantage. Instead, the greater core count of the RTX 6000 Ada proves more beneficial for models like NRX.

As such, the RTX 6000 Ada offers a more cost-effective and latency-efficient solution compared to the A100 for deploying neural receivers.


# Conclusions
{% include figure.html path="assets/img/2025-05-30-efficient-neural-receivers/Image_BER.png" class="img-fluid rounded z-depth-1" %}


