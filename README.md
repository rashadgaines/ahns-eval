**NOTE: These views are solely my own and do not reflect the opinions or influence of my employer, xAI.**

# AHNS Eval: Measuring Grok 2 Image Generation Model on Aesthetic Harmony and Novelty.


Generative models have revolutionized image creation, but evaluating their aesthetic quality remains challenging. This research introduces the AHNS framework to assess both harmony(aesthetics) and novelty(creatibity) in generated images, with a focus on X.AI's image gen model.

The AHNS rewards outputs that are aesthetically cohesive and creatively distinct, penalizing those that lack artistic harmony or are overly similar to training data or prior outputs.

## Methodology

### Data Collection

Images were generated using X.AI's Grok model via the OpenAI Python client. Prompts included diverse artistic scenarios, such as serene landscapes, abstract patterns, and surreal dreamscapes.

### Evaluation Metrics

1. **AHNS Score**: Combines harmony and novelty into a single metric.

   $AHNS = \frac{1}{N}\sum_{i=1}^{N} (w_h \cdot h_i + w_n \cdot n_i)$

   Where:
   - $N$ is the total number of images evaluated
   - $w_h$ and $w_n$ are weights for harmony and novelty (typically 0.5 each)
   - $h_i$ is the harmony score for image $i$
   - $n_i$ is the novelty score for image $i$

2. **Harmony Score**: Measures color harmony and composition.

   $H = \frac{1}{M}\sum_{i=1}^{M} \min_{s \in S} |h_i - s|$

   Where:
   - $M$ is the number of color harmony rules evaluated
   - $S$ is the set of ideal harmony values
   - $h_i$ represents the current image's harmony value
   - $\min_{s \in S}$ finds the closest ideal harmony value

3. **Novelty Score**: Quantifies uniqueness using embedding distances.

   $N = 1 - \frac{1}{K}\sum_{i=1}^{K} \cos(\theta_i)$

   Where:
   - $K$ is the number of reference images
   - $\theta_i$ is the angle between the current image's embedding and reference image $i$
   - $\cos(\theta_i)$ measures similarity (1 = identical, 0 = completely different)
   - The score is inverted (1 - mean) so higher values indicate more novelty

## Results

### Key Findings

- **AHNS Score**: 0.648 ± 0.096
- **Harmony Score**: 0.336 ± 0.192
- **Novelty Score**: 0.960 ± 0.000

#### Score Distribution
![Score Distribution](visualization/plots/score_distributions.png)
*Distribution of image quality scores showing Overall Quality (AHNS), Aesthetic Harmony, and Creative Novelty metrics. Each plot includes mean values (dashed lines) and standard deviations.*




## Citation

If you use this research in your work, please cite:
```
@software{ahns_eval,
  author = {Rashad Gaines},
  title = {AHNS Eval: Measuring Grok 2 Image Gen Model on Aesthetic Harmony and Novelty.},
  year = {2025},
  url = {https://github.com/rashadgaines/ahns-eval}
}
```