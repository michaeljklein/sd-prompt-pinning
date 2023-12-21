
# sd-prompt-pinning

Pin a prompt to a visual target!

An extension for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
based on:
- [DEAP](https://github.com/DEAP/deap) for optimization algorithms
- [ꟻLIP](https://github.com/NVlabs/flip) as a basis for a custom loss function


## Dependencies

This extension depends on
[https://github.com/picobyte/stable-diffusion-webui-wd14-tagger](picobyte/stable-diffusion-webui-wd14-tagger)
for image tagging, but it could possibly be made into an optional dependency.
(If it's an issue for your use case, I expect it to be straightforward to disable
in the script file as it's non-essential.)


## Problem

- Variation in prompts is hard to “pin down”: it can be difficult to tell which parts of the prompts are “locking in” a particular result.
  For example, a highly-specified prompt can produce results with little variation, even at lower CFG scales.
- Why is this useful?
    - Analyze larger prompts to tell which parts are “tighter” or “looser,” relative to a particular model, VAE, etc.
    - Refine precise prompts by eliminating certain variations.
    - Build “prompt pieces” for specifying particular behavior. E.g. prompt-based “bad hand” or “tarot card” embeddings.
    - Advanced:
        - Target images provide a simple way to pin to a particular image (i.e. for animation)
    - Unimplemented (at time of writing):
        - Target images that ignore an image mask, e.g. fix parts of an image for animation, solely using the prompt!
        - CLIP-based analysis to allow pinning a result to a particular (set of) goal tag(s)
          + Will need to add the following to `metadata.ini`
          + `Requires = stable-diffusion-webui-tokenizer`


## Solution

CMA (covariant matrix adaptation) is an efficient automatic evolutionary optimization method.
- It’s fit for problems where the input is a matrix and the metric is smooth.
- In practice, it converges exponentially.
- Downside(s):
    - Difficult to specify “small” distance from original prompt, so may need to use euclidean distance or similar.
        - This means that certain tokens could get “washed out” with larger allowed distances.
    - A sufficiently-large sample is required _per attempt_.
      For many cases, `8-16` images are likely sufficient, but assuming efficiency of “perfect” binary search,
      it will require around `3*num_tokens` steps to converge, or `3*num_tokens*batch_size` images.
        - By the way, binary search is about as efficient as stable diffusion:
          a few manual experiments showed that `2^steps` is approximately `bits_of_output`
          for "good" convergence, at least with `DPM++ SDE Karras`.


## Guide

### Options

Parameter	                   | Default	                                               | Details
---------------------        | ------------------------------------------------------- | -----------------------------------------------------------------------------------------------------------------
Target Images                | `None`                                                  | Use the provided image(s) as a target instead of the first generated batch
CMA Logging                  | `True`                                                  | Log CMA info to CLI (stdout)
CMA Seed                     | `[calculated from seed, subseed]`                       | Numpy seed, used for CMA sampling
Number of generations        | `int(16 * floor(log(N)))`                               | Number of generations
Initial population STD       | `0.05`                                                  | CMA initial population STD
Initial population radius    | `0.25`                                                  | Radius of uniform distribution for CMA initial population
Multi-objective size limiter | `0`                                                     | Disabled when `0`. Apply a penalty using a multi-objective CMA when more than this distance from original prompt
Size limit error             | `[size limiter] / 100`                                  | Error for multi-objective size limiter: vectors within this distance are "close"
Size limit weight            | `[size limiter] * 10`                                   | Weight for multi-objective size limiter penalty
`lambda_`                    | `int(4 + 3 * log(N))`                                   | Number of children to produce at each generation, `N` is the individual's size (integer).
`mu`                         | `int(lambda_ / 2)`                                      | The number of parents to keep from the lambda children (integer).
`cmatrix`                    | `identity(N)`                                           | The initial covariance matrix of the distribution that will be sampled.
`weights`                    | `"superlinear"`                                         | Decrease speed, can be `"superlinear"`, `"linear"` or `"equal"`.
`cs`                         | `(mueff + 2) / (N + mueff + 3)`                         | Cumulation constant for step-size.
`damps`                      | `1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs`  | Damping for step-size.
`ccum`                       | `4 / (N + 4)`                                           | Cumulation constant for covariance matrix.
`ccov1`                      | `2 / ((N + 1.3)^2 + mueff)`                             | Learning rate for rank-one update.
`ccovmu`                     | `2 * (mueff - 2 + 1 / mueff) / ((N + 2)^2 + mueff)`     | Learning rate for rank-mu update.

NOTE: Some parameters may not work when multi-objective size limiting is enabled.
The allowed parameters when multi-objective optimization are as follows:
(Some options may not yet be available in the UI.)

 Parameter    | Default                 | Details                   
------------- | ----------------------- | ------------------------------------------------
 `mu`         | `len(population)`       | The number of parents to use in the evolution. 
------------- | ----------------------- | ------------------------------------------------
 `lambda_`    | `1`                     | Number of children to produce at each generation 
------------- | ----------------------- | ------------------------------------------------
 `d`          | `1.0 + N / 2.0`         | Damping for step-size.    
------------- | ----------------------- | ------------------------------------------------
 `ptarg`      | `1.0 / (5 + 1.0 / 2.0)` | Target success rate.      
------------- | ----------------------- | ------------------------------------------------
 `cp`         | `ptarg / (2.0 + ptarg)` | Step size learning rate.  
------------- | ----------------------- | ------------------------------------------------
 `cc`         | `2.0 / (N + 2.0)`       | Cumulation time horizon.  
------------- | ----------------------- | ------------------------------------------------
 `ccov`       | `2.0 / (N**2 + 6.0)`    | Covariance matrix learning
              |                         | rate.                     
------------- | ----------------------- | ------------------------------------------------
 `pthresh`    | `0.44`                  | Threshold success rate.   
------------- | ----------------------- | ------------------------------------------------


Ref. `Hansen and Ostermeier, 2001. Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation`

Because the `Number of generations` lacks a default in the original implementation,
the default was picked from the following observances:
- When the algorithm is efficient, the number of generations is proportional to `log(N)`
- From the example [`cma_minfct`](https://github.com/DEAP/deap/blob/master/examples/es/cma_minfct.py)
  and by eyeballing other examples, the multiplicative overhead is approximately `16` (when the algorithm is efficient)


### Outputs

Assuming `txt2img` (it works similarly for `img2img`):

- `outputs/txt2img-images/prompt_pins`: all prompt pin runs
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number`: a particular prompt pin run
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/cma_plot.png`: CMA algorithm stats
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/index.html`: all stats summary webpage
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/00generation_number`: a particular generation within the prompt pin run
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/00generation_number/ijdwfemknbidwjo..`: a particular individual (attempt) within a generation
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/00generation_number/ijdwfemknbidwjo../20..-..-..`: a particular individual's image output
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/00generation_number/ijdwfemknbidwjo../batch_stats.json`: JSON of the batch's config and visual errors
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/00generation_number/ijdwfemknbidwjo../loss_plot.png`: PNG plot of visual errors
- `outputs/txt2img-images/prompt_pins/00prompt_pin_number/00generation_number/ijdwfemknbidwjo../summary.gif`: GIF of all images from the individual

### Techniques

If no target image is used:

1. Find a target prompt
2. Use X/Y/Z plot to hone in on number of steps, CFG scale, sampler, VAE, etc.
3. Plot larger batches until it has the desired amount of variation:
  - We want to find a batch size with many "good" results and visible variation (to "pin" down)
4. Use prompt pinning script
  - Ideally, it works with default arguments
  - If not, look at the `cma_plot.png` in the `txt2img-image/prompt_pins` folder for your run
    + If the upper left graph is branching out (like a `<` shape):
      * The generation size (i.e. batch count times batch size) is too small for how much error you have.
      * This could be because: your CFG scale is too small, the model is having trouble matching your prompt, or otherwise you have too much variation in your prompt
    + If the upper left graph is converging (like a `>` shape):
      * Try running for a larger number of generations, unless:
        - If the `>` shape is followed by a long line (like `>----` or similar), you've achieved convergence!
        - Try lowering the initial population centroid radius and STD: it could be that the CMA algorithm is searching too far from your prompt
5. Use the descovered "pinned" prompt in other prompts!

If a target image is used, a similar approach may be effective, but it's likely
that the generated images will need to be fairly close to the target image(s)
provided for good results. Otherwise, it may end up finding color or image
arrangement similarities to optimize.

Likewise, if visually-distinct target images are used, the algorithm is
effectively finding the "visual average," which is likely to be blurry,
distorted, or otherwise indistinct.

#### Debugging

- Upper right graph of `cma_plot.png` shows divergence
  + It's likely that it's not sampling "wide" enough, or is way too wide:
    * If way too wide, try lowering the initial population radius and STD
    * If not wide enough, try increasing the CFG scale, batch size, or `lambda_`


## Test runs (in progress.. 🚧)

[sd-prompt-pinning-test-cases](https://github.com/michaeljklein/sd-prompt-pinning-test-cases)

## References

- [DEAP](http://goo.gl/amJ3x)
  + Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner,
    Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy",
    Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012.
- [LDR ꟻLIP](https://research.nvidia.com/publication/2020-07_FLIP)

