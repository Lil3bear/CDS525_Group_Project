# Presentation Outline

## Slide 1: Title And Project Goal

- **Key bullet points**
- KMNIST Handwriting Recognition Using PyTorch
- Group members and course information
- Project goal: build the simplest reliable high-scoring model in one day
- **Figure(s)**
- None
- **Speaker notes**
- Introduce the project as a practical image-classification assignment.
- State clearly that the focus was not maximum complexity, but a strong result with a simple, explainable system.
- **Suggested time**
- 0.5 to 0.75 minute

## Slide 2: Problem Background

- **Key bullet points**
- Handwriting recognition is a computer vision classification task
- KMNIST contains 10 classes of Kuzushiji characters
- The dataset is harder than ordinary MNIST because several classes look visually similar
- **Figure(s)**
- None
- **Speaker notes**
- Explain why handwritten characters are a good deep learning problem for a student project.
- Briefly connect the task to real applications such as document analysis and digital archiving.
- **Suggested time**
- 1 minute

## Slide 3: Dataset And Preprocessing

- **Key bullet points**
- Training set: 60,000 images, test set: 10,000 images
- Image format: grayscale `28 x 28`
- Loaded from local `.npz` files in `data/KMNIST/npz`
- Training set split into 54,000 train and 6,000 validation samples
- Pixel scaling to `[0, 1]`, then normalization to approximately `[-1, 1]`
- **Figure(s)**
- None
- **Speaker notes**
- Keep this slide factual and quick.
- Mention that local `.npz` loading was used because the remote host was not reachable, but the dataset content is still KMNIST.
- **Suggested time**
- 1.25 minutes

## Slide 4: System Design And Project Functions

- **Key bullet points**
- `data.py`: data loading and train/validation/test preparation
- `model.py`: simple CNN definition
- `train.py`: training, evaluation, metric saving, checkpoint saving
- `experiments.py`: required comparisons
- `plots.py`: figure generation from CSV results
- `visualize_predictions.py`: first 100 test predictions
- **Figure(s)**
- None
- **Speaker notes**
- Emphasize that the codebase is flat and easy to explain.
- This helps show that the project was designed for clarity and reproducibility, not unnecessary abstraction.
- **Suggested time**
- 1 minute

## Slide 5: CNN Architecture And Training Setup

- **Key bullet points**
- Two convolution blocks with ReLU and max pooling
- Fully connected layer with 128 hidden units and dropout `0.25`
- Output layer with 10 classes
- Baseline settings: cross-entropy loss, Adam, learning rate `0.001`, batch size `64`, `12` epochs
- Fixed seed `42`, CPU execution in the completed runs
- **Figure(s)**
- None
- **Speaker notes**
- Explain why CNN is appropriate for images: it learns local visual patterns such as strokes and edges.
- State that the chosen setup was strong enough for the assignment while still simple enough to implement and justify in one day.
- **Suggested time**
- 1.5 minutes

## Slide 6: Baseline Result

- **Key bullet points**
- Best validation accuracy: `98.60%` at epoch 10
- Best test accuracy: `95.63%`
- Final test accuracy at epoch 12: `95.59%`
- Training loss decreased from `0.3538` to `0.0163`
- The model learned quickly and remained stable after the early epochs
- **Figure(s)**
- `figures/baseline_training_curves.png`
- **Speaker notes**
- Walk through the three curves: training loss, training accuracy, and test accuracy.
- Point out that the baseline alone already provides a strong assignment result.
- **Suggested time**
- 1.5 minutes

## Slide 7: Loss Function Comparison

- **Key bullet points**
- Baseline: `CrossEntropyLoss`
- Comparison: `CrossEntropyLoss(label_smoothing=0.1)`
- Label smoothing improved best validation accuracy from `98.60%` to `98.92%`
- Final test accuracy improved from `95.59%` to `96.40%`
- **Figure(s)**
- `figures/loss_comparison.png`
- **Speaker notes**
- Explain label smoothing in simple terms: it reduces overconfidence during training.
- Make the point that a small change in the loss function improved generalization without changing the CNN itself.
- **Suggested time**
- 1.25 minutes

## Slide 8: Learning-Rate Comparison

- **Key bullet points**
- Tested learning rates: `0.1`, `0.01`, `0.001`, `0.0001`
- `0.001` gave the best overall result
- `0.1` failed and stayed near chance level
- `0.01` learned but underperformed badly
- `0.0001` was stable but slower and less accurate within 12 epochs
- **Figure(s)**
- `figures/lr_comparison_loss.png`
- `figures/lr_comparison_accuracy.png`
- **Speaker notes**
- This is one of the most important analysis slides.
- Emphasize that learning rate mattered much more than batch size in this project.
- **Suggested time**
- 1.5 minutes

## Slide 9: Batch-Size Comparison

- **Key bullet points**
- Tested batch sizes: `8`, `16`, `32`, `64`, `128`
- All settings gave strong results around `95.5%` to `95.7%` best test accuracy
- Best result in the sweep: batch size `32` with `95.72%`
- Baseline batch size `64` remained a practical and reasonable default
- **Figure(s)**
- `figures/batch_comparison_loss.png`
- `figures/batch_comparison_accuracy.png`
- **Speaker notes**
- Explain that the batch-size effect was relatively small compared with the learning-rate effect.
- This supports the choice of keeping a standard batch size in a one-day project.
- **Suggested time**
- 1.5 minutes

## Slide 10: Qualitative Prediction Visualization

- **Key bullet points**
- Visualization uses the saved baseline checkpoint
- Displays the first 100 test samples with predicted and true labels
- The baseline model correctly predicted 96 of the first 100 test samples
- Qualitative inspection helps reveal both strengths and remaining errors
- **Figure(s)**
- `figures/first_100_test_predictions.png`
- **Speaker notes**
- Use this slide to make the model behavior visually concrete.
- Mention that most predictions are correct, but some mistakes remain on difficult or visually similar characters.
- **Suggested time**
- 1.5 minutes

## Slide 11: Conclusion, Limitations, And Future Work

- **Key bullet points**
- A simple CNN achieved strong KMNIST performance with minimal complexity
- Best baseline test accuracy: `95.63%`
- Best alternate-loss test accuracy: `96.40%`
- Main lesson: careful hyperparameter choice matters more than model complexity
- Future work: data augmentation, confusion matrix, more seeds, deeper CNN, learning-rate scheduling
- **Figure(s)**
- None
- **Speaker notes**
- End with a concise takeaway: the project met the assignment goals with a clear and reproducible solution.
- Keep the future work realistic and student-appropriate rather than overpromising.
- **Suggested time**
- 1 to 1.25 minutes

## Timing Advice

- Aim for about 13.5 to 14 minutes of speaking time and keep at least 1 minute as buffer.
- Do not read every number from the slides. Highlight only the key comparisons.
- Let one speaker handle the method slides and another handle the results slides to keep transitions clean.
- If time becomes tight, shorten Slides 3 and 4 first, not the results slides.
