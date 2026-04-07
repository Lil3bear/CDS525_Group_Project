# KMNIST Handwriting Recognition Using a Simple Convolutional Neural Network in PyTorch

## 1. Introduction

Handwriting recognition is a classic and still highly relevant problem in pattern recognition and computer vision. It is important in document analysis, form processing, digital archiving, postal automation, and the development of assistive technologies. Even though modern large-scale vision systems can solve much more complex tasks, handwritten character recognition remains a useful educational problem because it contains the core challenges of image classification in a compact form. A model must learn visual patterns, distinguish similar classes, and generalize to unseen samples, while also remaining efficient enough to train in a limited time.

This project focuses on KMNIST, a dataset of cursive Japanese characters derived from Kuzushiji. KMNIST is often described as a more challenging alternative to the standard MNIST digit dataset because the visual differences between classes are less obvious and some character categories are structurally similar. Each image is grayscale and has a resolution of 28 by 28 pixels, which makes the dataset small enough for fast experimentation but still meaningful for evaluating a deep learning model. The project goal was not to build the most complex model possible. Instead, the goal was to build the simplest reliable system that could achieve strong performance within one day while still covering the assignment requirements in a clear and reproducible way.

The implementation was completed in PyTorch with a flat and readable project structure. The finished system includes data loading, model definition, training, controlled experiments, metric saving, figure generation, checkpoint saving, and qualitative prediction visualization. The main model is a small convolutional neural network (CNN), and its training behavior is recorded across epochs. In addition, the project compares two loss-function settings, four learning rates, and five batch sizes, then generates report-ready figures from the saved metrics. A visualization of the first 100 test predictions is also produced to provide a qualitative view of model behavior.

In this repository, the dataset is loaded from local NumPy `.npz` files placed in `data/KMNIST/npz`. This does not change the learning task itself; it only changes the data access method used by the project. The loaded arrays still represent the same KMNIST training and test sets, and keeping them locally makes the workflow straightforward and reproducible within the project environment.

The main purpose of the report is to explain how the completed system was designed, why the chosen settings were reasonable for a one-day student project, and how the model performed in the completed experiments. The report uses the actual generated outputs in the repository, including:

- `figures/baseline_training_curves.png`
- `figures/loss_comparison.png`
- `figures/lr_comparison_accuracy.png`
- `figures/lr_comparison_loss.png`
- `figures/batch_comparison_accuracy.png`
- `figures/batch_comparison_loss.png`
- `figures/first_100_test_predictions.png`

The central question of the project can be stated simply: can a compact CNN, trained with a small number of sensible hyperparameter choices, achieve high KMNIST recognition accuracy and provide enough experimental evidence to support a strong university assignment submission? Based on the completed runs, the answer is yes. The final project produced a strong baseline, meaningful hyperparameter comparisons, and a clear set of figures suitable for a written report and presentation.

## 2. Design & Functions

The project was intentionally designed to remain simple. The codebase is organized into a small set of root-level files: `data.py`, `model.py`, `train.py`, `experiments.py`, `plots.py`, and `visualize_predictions.py`. This structure is appropriate for a student assignment because each file has one clear responsibility. `data.py` loads and prepares the KMNIST arrays, `model.py` defines the CNN, `train.py` handles the training and evaluation loop, `experiments.py` runs the required comparisons, `plots.py` generates figures from saved CSV files, and `visualize_predictions.py` creates the first-100 prediction grid. This layout avoids unnecessary abstraction and makes the workflow easy to explain during a demonstration.

### Dataset and preprocessing

The local dataset consists of four NumPy files:

- `kmnist-train-imgs.npz`
- `kmnist-train-labels.npz`
- `kmnist-test-imgs.npz`
- `kmnist-test-labels.npz`

The training image file contains 60,000 grayscale images with shape `(60000, 28, 28)`, and the test image file contains 10,000 grayscale images with shape `(10000, 28, 28)`. The labels are one-dimensional arrays with 60,000 and 10,000 elements respectively. In the data loader, the images are converted to floating-point tensors and reshaped into the CNN-compatible format `(N, 1, 28, 28)`, where the extra channel dimension represents grayscale input.

Before training, pixel values are scaled from the original `uint8` range into `[0, 1]` by dividing by 255 when necessary. After that, the project applies normalization using mean `0.5` and standard deviation `0.5`, which maps the images approximately into `[-1, 1]`. This is a common and simple preprocessing choice. It does not depend on heavy feature engineering, and it helps the network train more stably by keeping the input scale consistent.

The original training set is split into a training subset and a validation subset using a 90/10 ratio. This produces 54,000 training samples and 6,000 validation samples. The test set remains separate and is used only for reporting generalization performance. A fixed random seed of 42 is used for reproducibility, so the train-validation split remains stable across repeated runs. This setup is appropriate for the assignment because it allows the model to be tuned and compared using validation accuracy while still preserving an untouched test set for final evaluation.

### CNN architecture

The main model is a compact convolutional neural network defined in `model.py`. Its structure is:

1. Convolution layer from 1 channel to 32 channels, kernel size 3, padding 1
2. ReLU activation
3. Max pooling with kernel size 2
4. Convolution layer from 32 channels to 64 channels, kernel size 3, padding 1
5. ReLU activation
6. Max pooling with kernel size 2
7. Flatten
8. Fully connected layer from `64 x 7 x 7` to 128
9. ReLU activation
10. Dropout with probability 0.25
11. Final fully connected layer from 128 to 10 output classes

This architecture is simple but suitable for handwritten image classification. The convolution layers learn local stroke patterns, corners, and curves. Pooling reduces spatial size and helps the network focus on more abstract features. The fully connected layers then combine these learned features to produce class predictions. Because KMNIST images are small and grayscale, a very deep network would be unnecessary for this assignment. The selected CNN is strong enough to learn meaningful visual structure while still being lightweight enough to train many times on CPU within a reasonable amount of time.

### Loss function, optimizer, and hyperparameters

The main baseline uses `CrossEntropyLoss`, which is the standard loss function for multi-class image classification. This is a natural choice because each image belongs to exactly one of ten classes. Cross-entropy is easy to explain, widely used, and directly compatible with raw logits from the final linear layer.

The alternate loss experiment uses `CrossEntropyLoss(label_smoothing=0.1)`. Label smoothing slightly softens the target distribution instead of forcing the model to place all probability mass on a single class during training. This can reduce overconfidence and sometimes improve generalization. It is a good comparison because it changes only the loss behavior while leaving the model and optimizer unchanged.

The optimizer is Adam with a baseline learning rate of `0.001`. Adam was chosen because it converges quickly, usually needs less manual tuning than basic stochastic gradient descent, and is well suited to a small one-day project. The baseline batch size is `64`, and the baseline training length is `12` epochs. These values were selected to balance speed, stability, and assignment coverage. Twelve epochs are enough to show clear learning curves while keeping the total training time manageable across all required experiments.

The baseline configuration is therefore:

- Loss: `CrossEntropyLoss`
- Optimizer: Adam
- Learning rate: `0.001`
- Batch size: `64`
- Epochs: `12`
- Seed: `42`
- Device used in the completed runs: CPU

### Why this design suits a one-day student project

This design is suitable for a one-day university project for several reasons. First, the code is easy to understand. Second, the CNN is strong enough to achieve high performance without requiring advanced components such as residual connections, schedulers, or heavy augmentation. Third, the experiment plan is directly tied to the assignment requirements. Fourth, every run saves metrics and checkpoints automatically, which makes the results easy to trace and plot. Finally, the approach is reproducible: the seed, hyperparameters, output files, and figure generation process are all explicit.

In short, the project emphasizes clarity over novelty. That is the right tradeoff for a student assignment where the quality of explanation and completeness of experimentation matter as much as raw accuracy.

## 3. Demonstration & Performance

### Baseline training result

The baseline run is recorded in `results/baseline_metrics.csv` and summarized in `results/baseline_config.json`. Its learning curves are shown in `figures/baseline_training_curves.png`. The model learned quickly during the first few epochs and then improved more gradually. Training loss decreased from `0.3538` in epoch 1 to `0.0163` in epoch 12. Training accuracy increased from `89.01%` to `99.48%`, showing that the model successfully fit the training data.

Validation and test performance were also strong. The best validation accuracy was `98.60%` at epoch 10. The best observed test accuracy during the run was `95.63%`, and the final test accuracy at epoch 12 was `95.59%`. These numbers show that the model generalized well despite being compact. The gap between training accuracy and test accuracy indicates some degree of overfitting, which is expected for a small CNN trained on a finite dataset, but the generalization level remained high enough for the assignment.

The baseline result is important because it provides a stable reference point for the later comparisons. It demonstrates that a straightforward CNN with standard cross-entropy loss and Adam optimization can already solve KMNIST at a high level of accuracy without complicated engineering. This supports the project goal of building the simplest high-scoring solution that can be completed in one day.

### Alternate loss comparison

The loss comparison is shown in `figures/loss_comparison.png`. In this experiment, the baseline cross-entropy setting was compared with a label-smoothed cross-entropy variant using the same architecture, optimizer, learning rate, batch size, and number of epochs. This makes the comparison fair because the only intended difference is the loss behavior.

The alternate loss achieved stronger performance than the plain baseline. According to `results/loss_label_smoothing_config.json`, its best validation accuracy was `98.92%` at epoch 10, compared with `98.60%` for the baseline. Its best test accuracy reached `96.40%`, and its final test accuracy was also `96.40%`. In other words, label smoothing improved the final test result by about `0.81` percentage points over the baseline final test accuracy of `95.59%`.

This is a meaningful finding for the report. It suggests that the baseline model was already strong, but a small adjustment to the loss function helped the network generalize slightly better. A likely explanation is that label smoothing reduced excessive confidence on the training data and encouraged softer decision boundaries. Even though the project kept plain cross-entropy as the main baseline because it is the most standard and simplest option to explain, the comparison shows that regularized classification losses can provide a modest improvement without changing the model architecture.

### Learning-rate comparison

The learning-rate comparison is shown in `figures/lr_comparison_loss.png` and `figures/lr_comparison_accuracy.png`, with the summary recorded in `results/lr_sweep_summary.csv`. The tested values were `0.1`, `0.01`, `0.001`, and `0.0001`, while all other settings remained the same as the baseline.

The results clearly show that the learning rate was one of the most important hyperparameters in the project:

- `0.1` failed almost completely. Its best validation accuracy was only `10.50%`, and test accuracy stayed at `10.00%`, which is essentially chance level for a 10-class problem. The learning rate was too large, causing unstable updates and preventing meaningful learning.
- `0.01` learned somewhat, but it still performed much worse than the baseline. Its best validation accuracy was `95.02%`, and its best test accuracy was `87.70%`, with final test accuracy `87.04%`. This value was lower than ideal and likely caused optimization to overshoot good minima.
- `0.001` produced the best overall performance among the tested learning rates. Its best validation accuracy was `98.60%`, and its best test accuracy was `95.63%`.
- `0.0001` was more stable than the overly large rates, but it learned more slowly. Its best validation accuracy was `97.92%`, and final test accuracy was `94.02%`, which is good but still below `0.001`.

These outcomes are easy to interpret. If the learning rate is too high, the optimizer cannot settle into a good solution. If it is too low, learning becomes slow and may not reach the best performance within the fixed 12-epoch budget. In this project, `0.001` gave the best balance between speed and convergence, which is consistent with common practice when using Adam.

### Batch-size comparison

The batch-size comparison is shown in `figures/batch_comparison_loss.png` and `figures/batch_comparison_accuracy.png`, with the summary recorded in `results/batch_sweep_summary.csv`. The tested batch sizes were `8`, `16`, `32`, `64`, and `128`, with the learning rate fixed at `0.001`.

Compared with the learning-rate study, the batch-size study showed much smaller performance differences. All tested batch sizes produced strong results:

- Batch size `8`: best test accuracy `95.70%`
- Batch size `16`: best test accuracy `95.70%`
- Batch size `32`: best test accuracy `95.72%`
- Batch size `64`: best test accuracy `95.63%`
- Batch size `128`: best test accuracy `95.54%`

The highest best test accuracy in this sweep was obtained by batch size `32`, but the margin over the other strong settings was very small. This suggests that the CNN is fairly robust to batch size under the current problem scale and training budget. Since batch size `64` was already near the top and is a common practical default, keeping it as the baseline remains defensible. It offers a good compromise between computational efficiency and stable optimization. Batch size `128` was slightly weaker, possibly because the larger batches produced less noisy but less exploratory updates. Smaller batches such as `8` and `16` worked well, but they would generally take longer per epoch in practical use.

For a student report, this comparison is useful because it shows that not every hyperparameter has an equally dramatic effect. The project demonstrates that learning rate mattered much more than batch size for this model on this dataset.

### Qualitative prediction visualization

The qualitative result is shown in `figures/first_100_test_predictions.png`, which displays the first 100 test samples together with their predicted and true labels. A read-only check with the saved baseline checkpoint showed that the model predicted `96` of the first `100` test samples correctly. This aligns well with the overall test accuracy of roughly `95%` to `96%`.

This figure is valuable because it complements the numeric metrics. Accuracy values summarize performance, but they do not show what correct and incorrect predictions look like. The first-100 grid makes the behavior more concrete. Most images are classified correctly, which indicates that the model has learned useful stroke and shape patterns. The mistakes that remain are likely concentrated among visually similar or ambiguous characters, which is expected for KMNIST. For a university presentation, this figure is especially helpful because it gives the audience an immediate visual understanding of the model’s practical behavior.

### Overall discussion

Taken together, the experiments support several conclusions. First, the baseline design was already strong and suitable for the assignment. Second, the most important hyperparameter in this project was the learning rate. Third, a more careful loss function such as label smoothing can yield a measurable improvement without increasing architectural complexity. Fourth, batch size had a relatively small effect compared with learning rate.

The results also show a good balance between simplicity and performance. The model is small, the data preprocessing is minimal, and the code is easy to explain. Yet the project still achieved a baseline best test accuracy of `95.63%` and an alternate-loss best test accuracy of `96.40%`. For a one-day student project, this is a strong outcome.

### Limitations and future work

Although the project met its goals, it still has limitations. The CNN is intentionally simple, so there may be additional accuracy available from deeper architectures or more advanced regularization. The project also used a fixed 12-epoch budget without early stopping or learning-rate scheduling. These additions might improve efficiency or final accuracy. Another limitation is that the experiments were run with a single random seed, so variance across seeds was not studied. A more complete evaluation could include repeated runs and averages.

The project also does not include data augmentation, confusion-matrix analysis, or class-wise performance breakdown. These would be useful next steps if more time were available. Augmentation such as small translations or rotations might improve robustness. A confusion matrix could reveal which Kuzushiji characters are most often mixed up. In addition, a small comparison against a multilayer perceptron or a deeper CNN could strengthen the report by showing why convolution is the better design choice for image inputs.

Despite these limitations, the project succeeded in its original aim. It produced a reliable, explainable, and high-performing handwriting recognition system within a short time frame, and it generated all the evidence needed for a written report and oral presentation.

## 4. Conclusion

This project developed and evaluated a PyTorch-based CNN for KMNIST handwriting recognition with the objective of creating the simplest high-scoring solution that could realistically be completed in one day. The final system was deliberately compact: local NumPy-based dataset loading, straightforward normalization, a two-block convolutional network, Adam optimization, controlled experiments, automatic metric logging, figure generation, and checkpoint saving.

The baseline configuration of cross-entropy loss, learning rate `0.001`, batch size `64`, and `12` epochs achieved a best validation accuracy of `98.60%` and a best test accuracy of `95.63%`. This already demonstrates that a simple CNN is highly effective on KMNIST. The additional experiments provided useful insight. Label smoothing improved test performance to `96.40%`. The learning-rate sweep showed that `0.001` was the most appropriate value in this setup, while `0.1` completely failed and `0.01` performed much worse than expected. The batch-size sweep showed that performance was robust across a wide range of values, with batch size `32` giving the highest best test accuracy at `95.72%`, although the difference from the baseline was small.

From an academic perspective, the project is successful because it does more than report a single accuracy number. It explains the design choices, compares alternatives, presents quantitative evidence through figures, and adds qualitative inspection through the first-100 prediction visualization. The result is a complete and defensible assignment submission rather than a simple coding exercise.

The most important lesson from the project is that strong performance does not always require a complicated model. Careful choice of a standard CNN, sensible preprocessing, a reproducible training setup, and a focused experiment plan were enough to achieve a strong result. For a student team working under time constraints, this is a practical and valuable outcome.

## 5. References

- [Ref 1] KMNIST / Kuzushiji-MNIST dataset source and description.
- [Ref 2] PyTorch documentation for tensors, modules, optimization, and training workflows.
- [Ref 3] Torch / deep learning reference for convolutional neural networks in image classification.
- [Ref 4] Academic or course reference discussing cross-entropy loss and label smoothing.
- [Ref 5] Academic or course reference discussing the effect of learning rate and batch size on optimization.

Replace the placeholders above with the final course-approved references before submission.
