# Notebook to document changes to the code

- Added a normalization process so that the "resting" values of each articulator vary from [0, 1] to avoid any issues with regression.
- Added an additional articulator TC to represent frontness
- Adding masking in order to improve MSE loss function (prevent it from doing regression on articulators in resting state)